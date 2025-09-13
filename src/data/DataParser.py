import xml.etree.ElementTree as ET
import json
import zipfile
import csv
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re

class DataParser:
    """
    A class to parse MedQuAD XML files and extract Q&A data into a structured format.
    Includes answer recovery functionality for problematic folders.
    """
    
    def __init__(self, data_root: str = "data/raw/MedQuAD", verbose: bool = False):
        """
        Initialize the DataParser.
        
        Args:
            data_root (str): Root directory path for MedQuAD data
            verbose (bool): Enable verbose logging for debugging
        """
        self.data_root = Path(data_root)
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.parsed_data = {"unlabeled_qa_examples": []}
        self.answer_recovery_map = {}
        self.folders_needing_recovery = {"10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}
        self.question_counter = 0  # Simple incrementing counter for unique IDs
        
        # Risk level mapping based on question type
        self.question_type_risk_mapping = {
            # High Risk (3.0x penalty) 
            "treatment": "High Risk",
            "side effects": "High Risk",
            "emergency or overdose": "High Risk",
            "severe reaction": "High Risk",
            "contraindication": "High Risk",
            "usage": "High Risk",
            "forget a dose": "High Risk",
            "important warning": "High Risk",
            "indication": "High Risk",
            "precautions": "High Risk",
            
            # Medium Risk (2.0x penalty)
            "symptoms": "Medium Risk",
            "causes": "Medium Risk",
            "outlook": "Medium Risk",
            "considerations": "Medium Risk",
            "when to contact a medical professional": "Medium Risk",
            "complications": "Medium Risk",
            "exams and tests": "Medium Risk",
            "susceptibility": "Medium Risk",
            "storage and disposal": "Medium Risk",
            "brand names of combination products": "Medium Risk",
            "stages": "Medium Risk",
            "genetic changes": "Medium Risk",
            "inheritance": "Medium Risk",
            "frequency": "Medium Risk",
            
            # Low Risk (1.0x penalty) 
            "information": "Low Risk",
            "support groups": "Low Risk",
            "prevention": "Low Risk",
            "dietary": "Low Risk",
            "other information": "Low Risk",
            "brand names": "Low Risk",
            "why get vaccinated": "Low Risk",
            "how can i learn more": "Low Risk",
            "research": "Low Risk"
        }
        
        # Auto-load answer recovery data
        self._load_answer_recovery_data()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the parser."""
        logger = logging.getLogger(__name__)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
            # Console handler for verbose mode
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for detailed debugging
            file_handler = logging.FileHandler('data_parser_debug.log', mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        else:
            logger.setLevel(logging.INFO)
            # Console handler for normal mode - only important messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _extract_zip_if_needed(self) -> Optional[Path]:
        """
        Extract the answer recovery ZIP file if not already extracted.
        
        Returns:
            Optional[Path]: Path to the extracted CSV file, or None if not found
        """
        zip_path = self.data_root / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip"
        csv_path = self.data_root / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers" / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers" / "All-2479-Answers-retrieved-from-MedQuAD.csv"
        
        # Check if CSV already exists
        if csv_path.exists():
            if self.verbose:
                self.logger.info(f"Answer recovery CSV already exists: {csv_path}")
            return csv_path
        
        # Check if ZIP exists
        if not zip_path.exists():
            self.logger.warning(f"Answer recovery ZIP file not found: {zip_path}")
            return None
        
        try:
            self.logger.info(f"Extracting answer recovery data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_root)
            
            if csv_path.exists():
                self.logger.info(f"Successfully extracted answer recovery data")
                return csv_path
            else:
                self.logger.error(f"CSV file not found after extraction")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting ZIP file: {e}")
            return None
    
    def _load_answer_recovery_data(self) -> None:
        """
        Load answer recovery data from the CSV file.
        """
        csv_path = self._extract_zip_if_needed()
        if not csv_path:
            self.logger.warning("Answer recovery data not available")
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    answer_id = row.get('AnswerID', '').strip()
                    answer_text = row.get('Answer', '').strip()
                    
                    if answer_id and answer_text:
                        # Extract the answer part after "Answer:"
                        recovered_answer = self._extract_answer_from_recovery_text(answer_text)
                        if recovered_answer:
                            self.answer_recovery_map[answer_id] = recovered_answer
            
            self.logger.info(f"Loaded {len(self.answer_recovery_map)} answer recovery entries")
            
        except Exception as e:
            self.logger.error(f"Error loading answer recovery data: {e}")
    
    def _extract_answer_from_recovery_text(self, text: str) -> Optional[str]:
        """
        Extract the answer portion from the recovery text format.
        
        Args:
            text (str): Full recovery text containing question, URL, and answer
            
        Returns:
            Optional[str]: Extracted answer text, or None if not found
        """
        # Look for "Answer:" and extract everything after it
        answer_match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
        if not answer_match:
            return None
        
        answer_text = answer_match.group(1).strip()
        
        # Remove trailing parentheses and extra whitespace
        answer_text = re.sub(r'\s*\)\s*$', '', answer_text)
        answer_text = ' '.join(answer_text.split())
        
        return answer_text if answer_text else None
    
    def _get_recovery_key_from_xml_path(self, xml_path: Path, qa_pair_element) -> Optional[str]:
        """
        Generate a recovery key that might match entries in the CSV file.
        
        Args:
            xml_path (Path): Path to the XML file
            qa_pair_element: XML element containing the QA pair
            
        Returns:
            Optional[str]: Recovery key to look up in the answer recovery map
        """
        # Extract the base filename without extension
        base_filename = xml_path.stem
        
        # Get the QA pair ID
        pid = qa_pair_element.get('pid', '1')
        
        # Try different key formats that might be in the CSV based on the folder
        possible_keys = []
        
        if "10_MPlus_ADAM_QA" in str(xml_path):
            # ADAM format
            possible_keys.extend([
                f"ADAM_{base_filename}_Sec{pid}.txt",
                f"ADAM_{base_filename}_{pid}.txt",
                f"ADAM_{base_filename}.txt"
            ])
        elif "11_MPlusDrugs_QA" in str(xml_path):
            # MPlusDrugs format
            possible_keys.extend([
                f"MPlusDrugs_{base_filename}_Sec{pid}.txt",
                f"MPlusDrugs_{base_filename}_{pid}.txt",
                f"MPlusDrugs_{base_filename}.txt"
            ])
        elif "12_MPlusHerbsSupplements_QA" in str(xml_path):
            # MPlus format for herbs/supplements
            possible_keys.extend([
                f"MPlus_{base_filename}_Sec{pid}.txt",
                f"MPlus_{base_filename}_{pid}.txt",
                f"MPlus_{base_filename}.txt",
                f"MPlusHerbsSupplements_{base_filename}_Sec{pid}.txt",
                f"MPlusHerbsSupplements_{base_filename}_{pid}.txt",
                f"MPlusHerbsSupplements_{base_filename}.txt"
            ])
        
        # Also try generic formats
        possible_keys.extend([
            f"{base_filename}_Sec{pid}.txt",
            f"{base_filename}_{pid}.txt",
            f"{base_filename}.txt"
        ])
        
        # Check which key exists in our recovery map
        for key in possible_keys:
            if key in self.answer_recovery_map:
                if self.verbose:
                    self.logger.debug(f"Found matching key: {key} for file {xml_path.name}")
                return key
        
        return None
    
    def _get_next_question_id(self) -> str:
        """
        Generate the next unique question ID.
        
        Returns:
            str: Next question ID (Q1, Q2, Q3, etc.)
        """
        self.question_counter += 1
        return f"Q{self.question_counter}"
    
    def _get_risk_level(self, question_type: str) -> str:
        """
        Get risk level based on question type.
        
        Args:
            question_type (str): The question type
            
        Returns:
            str: Risk level (High Risk, Medium Risk, Low Risk)
        """
        # Convert to lowercase for case-insensitive matching
        question_type_lower = question_type.lower().strip()
        
        # Return mapped risk level or default to Medium Risk if not found
        return self.question_type_risk_mapping.get(question_type_lower, "Medium Risk")
    
    def parse_xml_file(self, file_path: Path) -> tuple[List[Dict[str, Any]], int]:
        """
        Parse a single XML file and extract Q&A pairs.
        
        Args:
            file_path (Path): Path to the XML file
            
        Returns:
            tuple[List[Dict[str, Any]], int]: List of extracted Q&A examples and count of recovered answers
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract document-level information
            doc_id = root.get('id', 'unknown')
            source = root.get('source', 'unknown')
            url = root.get('url', '')
            
            # Extract focus (medical topic)
            focus_element = root.find('Focus')
            medical_focus = focus_element.text if focus_element is not None else 'unknown'
            
            # Check if this file is from a folder that needs answer recovery
            needs_recovery = any(folder in str(file_path) for folder in self.folders_needing_recovery)
            
            # Extract Q&A pairs
            qa_examples = []
            recovered_count = 0
            qa_pairs = root.find('QAPairs')
            
            if qa_pairs is not None:
                qa_pair_list = qa_pairs.findall('QAPair')
                
                for i, qa_pair in enumerate(qa_pair_list):
                    pid = qa_pair.get('pid', str(i + 1))
                    
                    question_element = qa_pair.find('Question')
                    answer_element = qa_pair.find('Answer')
                    
                    if question_element is not None:
                        question_text = question_element.text or ''
                        question_type = question_element.get('qtype', 'information')
                        
                        # Clean up question text
                        question_text = ' '.join(question_text.split())
                        
                        # Handle answer text
                        answer_text = ''
                        answer_recovered = False
                        
                        if answer_element is not None and answer_element.text:
                            # Answer exists in XML
                            answer_text = ' '.join(answer_element.text.split())
                        elif needs_recovery:
                            # Try to recover answer from CSV data
                            recovery_key = self._get_recovery_key_from_xml_path(file_path, qa_pair)
                            if recovery_key and recovery_key in self.answer_recovery_map:
                                answer_text = self.answer_recovery_map[recovery_key]
                                answer_recovered = True
                                recovered_count += 1
                        
                        # Only include Q&A pairs that have both question and answer
                        if question_text and answer_text:
                            # Get risk level based on question type
                            risk_level = self._get_risk_level(question_type)
                            
                            qa_example = {
                                "id": self._get_next_question_id(),  # Simple incrementing ID
                                "question": question_text,
                                "answer": answer_text,
                                "question_type": question_type,
                                "medical_focus": medical_focus,
                                "source": source,
                                "risk_level": risk_level,
                                "processing_status": "unprocessed"
                            }
                            
                            # Add recovery status for debugging
                            if answer_recovered:
                                qa_example["_answer_recovered"] = True
                            
                            qa_examples.append(qa_example)
            
            # Only log recovery successes or verbose mode
            if recovered_count > 0 and self.verbose:
                self.logger.info(f"Recovered {recovered_count} answers from {file_path.name}")
                
            return qa_examples, recovered_count
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in {file_path}: {e}")
            return [], 0
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return [], 0
    
    def parse_directory(self, directory_path: Optional[Path] = None) -> None:
        """
        Parse all XML files in a directory and its subdirectories.
        
        Args:
            directory_path (Optional[Path]): Directory to parse. If None, uses self.data_root
        """
        if directory_path is None:
            directory_path = self.data_root
            
        if not directory_path.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return
        
        xml_files = list(directory_path.rglob("*.xml"))
        self.logger.info(f"Found {len(xml_files)} XML files to parse")
        
        total_examples = 0
        total_recovered = 0
        processed_files = 0
        
        for xml_file in xml_files:
            examples, recovered_count = self.parse_xml_file(xml_file)
            self.parsed_data["unlabeled_qa_examples"].extend(examples)
            
            total_examples += len(examples)
            total_recovered += recovered_count
            processed_files += 1
            
            # Progress indicator for large datasets
            if processed_files % 500 == 0:
                self.logger.info(f"Processed {processed_files}/{len(xml_files)} files...")
        
        self.logger.info(f"Parsing complete: {total_examples} Q&A pairs extracted")
        if total_recovered > 0:
            self.logger.info(f"Successfully recovered {total_recovered} missing answers")
    
    def parse_specific_subdirectories(self, subdirectories: List[str]) -> None:
        """
        Parse XML files from specific subdirectories.
        
        Args:
            subdirectories (List[str]): List of subdirectory names to parse
        """
        for subdir in subdirectories:
            subdir_path = self.data_root / subdir
            if subdir_path.exists():
                self.logger.info(f"Parsing subdirectory: {subdir}")
                self.parse_directory(subdir_path)
            else:
                self.logger.warning(f"Subdirectory not found: {subdir_path}")
    
    def get_parsed_data(self) -> Dict[str, Any]:
        """
        Get the parsed data in the required format.
        
        Returns:
            Dict[str, Any]: Parsed data in the specified format
        """
        return self.parsed_data
    
    def _create_database_table(self, cursor: sqlite3.Cursor) -> None:
        """
        Create the unlabeled_qa table if it doesn't exist.
        
        Args:
            cursor: SQLite cursor object
        """
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unlabeled_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                question_type VARCHAR NOT NULL,
                medical_focus VARCHAR,
                source VARCHAR,
                risk_level VARCHAR,
                processing_status VARCHAR DEFAULT 'unprocessed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    def save_to_database(self, db_path: str = "data/processed/unlabeled_qa.db", 
                        include_debug_info: bool = False) -> None:
        """
        Save the parsed data to a SQLite database.
        
        Args:
            db_path (str): Path where to save the SQLite database
            include_debug_info (bool): Whether to include debug information like recovery status
        """
        try:
            # Create the directory if it doesn't exist
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            self._create_database_table(cursor)
            
            # Clear existing data (optional - remove if you want to append)
            cursor.execute("DELETE FROM unlabeled_qa")
            
            # Prepare data for insertion
            examples = self.parsed_data["unlabeled_qa_examples"]
            
            # Insert data
            inserted_count = 0
            for example in examples:
                # Skip debug info unless requested
                if not include_debug_info and example.get("_answer_recovered"):
                    pass  # Still include the record, just don't mark it specially
                
                cursor.execute('''
                    INSERT INTO unlabeled_qa 
                    (question, answer, question_type, medical_focus, source, risk_level, processing_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    example["question"],
                    example["answer"],
                    example["question_type"],
                    example.get("medical_focus", ""),
                    example.get("source", ""),
                    example.get("risk_level", "Medium Risk"),
                    example.get("processing_status", "unprocessed")
                ))
                inserted_count += 1
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            self.logger.info(f"Data saved to SQLite database: {db_path}")
            self.logger.info(f"Total examples saved: {inserted_count}")
            
        except Exception as e:
            self.logger.error(f"Error saving to database {db_path}: {e}")
            if 'conn' in locals():
                conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the parsed data.
        
        Returns:
            Dict[str, Any]: Statistics about the parsed data
        """
        examples = self.parsed_data["unlabeled_qa_examples"]
        
        if not examples:
            return {"total_examples": 0}
        
        # Count by source, question type, risk level
        sources = {}
        question_types = {}
        risk_levels = {}
        medical_focuses = {}
        recovered_answers = 0
        
        for example in examples:
            source = example.get("source", "unknown")
            qtype = example.get("question_type", "unknown")
            risk = example.get("risk_level", "unknown")
            focus = example.get("medical_focus", "unknown")
            
            sources[source] = sources.get(source, 0) + 1
            question_types[qtype] = question_types.get(qtype, 0) + 1
            risk_levels[risk] = risk_levels.get(risk, 0) + 1
            medical_focuses[focus] = medical_focuses.get(focus, 0) + 1
            
            if example.get("_answer_recovered", False):
                recovered_answers += 1
        
        stats = {
            "total_examples": len(examples),
            "sources": sources,
            "question_types": question_types,
            "risk_levels": risk_levels,
            "unique_medical_focuses": len(medical_focuses),
            "top_medical_focuses": dict(sorted(medical_focuses.items(), 
                                             key=lambda x: x[1], reverse=True)[:10]),
            "answers_recovered": recovered_answers,
            "recovery_available": len(self.answer_recovery_map) > 0
        }
        
        return stats
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about answer recovery.
        
        Returns:
            Dict[str, Any]: Recovery statistics
        """
        recovery_stats = {
            "total_recovery_entries": len(self.answer_recovery_map),
            "folders_with_recovery": list(self.folders_needing_recovery),
            "recovery_available": len(self.answer_recovery_map) > 0
        }
        
        if self.answer_recovery_map:
            # Analyze recovery entries by source type
            recovery_by_source = {}
            for key in self.answer_recovery_map.keys():
                if key.startswith('ADAM_'):
                    source = 'ADAM'
                elif key.startswith('MPlusDrugs_'):
                    source = 'MPlusDrugs'
                elif key.startswith('MPlus'):
                    source = 'MPlus'
                else:
                    source = 'Other'
                
                recovery_by_source[source] = recovery_by_source.get(source, 0) + 1
            
            recovery_stats["recovery_entries_by_source"] = recovery_by_source
        
        return recovery_stats
    
    def debug_recovery_keys(self, sample_xml_files: List[str] = None) -> None:
        """
        Debug function to understand the key matching process.
        
        Args:
            sample_xml_files: List of XML filenames to debug, or None for automatic selection
        """
        print("=== RECOVERY KEY DEBUGGING ===")
        
        if not self.answer_recovery_map:
            print("No recovery data loaded!")
            return
        
        # Show recovery map statistics
        print(f"Total recovery entries: {len(self.answer_recovery_map)}")
        
        # Group keys by pattern
        patterns = {}
        for key in self.answer_recovery_map.keys():
            if key.startswith('ADAM_'):
                patterns.setdefault('ADAM', []).append(key)
            elif key.startswith('MPlusDrugs_'):
                patterns.setdefault('MPlusDrugs', []).append(key)
            elif key.startswith('MPlus'):
                patterns.setdefault('MPlus', []).append(key)
            else:
                patterns.setdefault('Other', []).append(key)
        
        for pattern, keys in patterns.items():
            print(f"{pattern} keys: {len(keys)} entries")
            print(f"  Sample: {keys[:3]}")
        
        print("=== END RECOVERY KEY DEBUGGING ===")
    
    def clear_data(self) -> None:
        """Clear the parsed data and reset counter."""
        self.parsed_data = {"unlabeled_qa_examples": []}
        self.question_counter = 0  # Reset counter when clearing data
        self.logger.info("Parsed data cleared and counter reset")


# Example usage
if __name__ == "__main__":
    # Initialize the parser with clean logging (set verbose=True for detailed debugging)
    parser = DataParser(verbose=False)
    
    # Get recovery statistics
    recovery_stats = parser.get_recovery_statistics()
    print("Answer Recovery Statistics:")
    print(json.dumps(recovery_stats, indent=2))
    
    # Parse all XML files in the MedQuAD directory
    parser.parse_directory()
    
    # Or parse specific subdirectories (including those needing recovery)
    # parser.parse_specific_subdirectories(['10_MPlus_ADAM_QA', '11_MPlusDrugs_QA', '12_MPlusHerbsSupplements_QA'])
    
    # Get parsing statistics
    # stats = parser.get_statistics()
    # print("\nParsing Statistics:")
    # print(json.dumps(stats, indent=2))
    
    # Save to SQLite database
    parser.save_to_database("data/processed/unlabeled_qa.db")
