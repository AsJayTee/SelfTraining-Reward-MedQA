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
    It also parses test questions and generates test_qna.json with risk levels
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
        
        # Enhanced risk level mapping based on question type - covers all 25 test question types
        self.question_type_risk_mapping = {
            # High Risk (3.0x penalty) - Critical medical decisions, treatments, drug interactions, dosing
            "treatment": "High Risk",
            "side effects": "High Risk",
            "side_effect": "High Risk",  # Test dataset format
            "emergency or overdose": "High Risk",
            "severe reaction": "High Risk",
            "contraindication": "High Risk",
            "usage": "High Risk",
            "forget a dose": "High Risk",
            "important warning": "High Risk",
            "indication": "High Risk",
            "precautions": "High Risk",
            "interaction": "High Risk",  # Drug interactions are dangerous
            "tapering": "High Risk",     # Medication tapering is critical
            "dosage": "High Risk",       # Medication dosing is critical
            
            # Medium Risk (2.0x penalty) - Medical assessment, diagnosis, symptoms, complications
            "symptoms": "Medium Risk",
            "symptom": "Medium Risk",    # Test dataset format
            "causes": "Medium Risk",
            "cause": "Medium Risk",      # Test dataset format
            "outlook": "Medium Risk",
            "considerations": "Medium Risk",
            "when to contact a medical professional": "Medium Risk",
            "complications": "Medium Risk",
            "complication": "Medium Risk", # Test dataset format
            "exams and tests": "Medium Risk",
            "susceptibility": "Medium Risk",
            "storage and disposal": "Medium Risk",
            "storage_disposal": "Medium Risk", # Test dataset format
            "brand names of combination products": "Medium Risk",
            "stages": "Medium Risk",
            "genetic changes": "Medium Risk",
            "inheritance": "Medium Risk",
            "frequency": "Medium Risk",
            "diagnosis": "Medium Risk",   # Medical assessment
            "effect": "Medium Risk",      # Medical effects
            "prognosis": "Medium Risk",   # Medical outlook
            "action": "Medium Risk",      # Medical actions
            "comparison": "Medium Risk",  # Comparing treatments/conditions
            "other_question": "Medium Risk", # Default for unknown medical questions
            
            # Low Risk (1.0x penalty) - Informational, educational, general guidance
            "information": "Low Risk",
            "support groups": "Low Risk",
            "prevention": "Low Risk",
            "dietary": "Low Risk",
            "other information": "Low Risk",
            "brand names": "Low Risk",
            "why get vaccinated": "Low Risk",
            "how can i learn more": "Low Risk",
            "research": "Low Risk",
            "ingredient": "Low Risk",     # Informational about ingredients
            "person_organization": "Low Risk", # Finding doctors/organizations
            "lifestyle_diet": "Low Risk", # General lifestyle advice
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
            Optional[Path]: Path to the extracted directory, or None if failed
        """
        zip_file_path = Path("data/raw/MedQuAD/QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip")
        
        if not zip_file_path.exists():
            self.logger.warning(f"ZIP file not found: {zip_file_path}")
            return None
            
        # Check if already extracted
        expected_extract_dir = zip_file_path.parent / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers"
        if expected_extract_dir.exists():
            self.logger.debug("ZIP file already extracted")
            return expected_extract_dir
        
        try:
            self.logger.info("Extracting answer recovery ZIP file...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(zip_file_path.parent)
            self.logger.info(f"Extracted ZIP file to: {expected_extract_dir}")
            return expected_extract_dir
        except Exception as e:
            self.logger.error(f"Failed to extract ZIP file: {e}")
            return None
    
    def _load_answer_recovery_data(self) -> None:
        """
        Load the answer recovery CSV file into a dictionary for quick lookups.
        """
        # Extract ZIP file if needed
        extract_dir = self._extract_zip_if_needed()
        if not extract_dir:
            self.logger.warning("Could not load answer recovery data - extraction failed")
            return
            
        # Try to find the CSV file
        csv_file_path = extract_dir / "All-2479-Answers-retrieved-from-MedQuAD.csv"
        
        if not csv_file_path.exists():
            self.logger.warning(f"Answer recovery CSV not found: {csv_file_path}")
            return
        
        try:
            self.answer_recovery_map = {}
            with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    answer_id = row.get('AnswerID', '').strip()
                    answer_text = row.get('Answer', '').strip()
                    
                    if answer_id and answer_text:
                        # Parse the answer to extract just the answer part
                        parsed_answer = self._extract_answer_from_recovery_text(answer_text)
                        if parsed_answer:
                            self.answer_recovery_map[answer_id] = parsed_answer
            
            self.logger.info(f"Loaded {len(self.answer_recovery_map)} answer recovery entries")
            
        except Exception as e:
            self.logger.error(f"Error loading answer recovery data: {e}")
    
    def _extract_answer_from_recovery_text(self, text: str) -> Optional[str]:
        """
        Extract the answer text from the full recovery text format.
        
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
            xml_path (Path): Path to the XML file being processed
            qa_pair_element: QAPair XML element
            
        Returns:
            Optional[str]: Recovery key to look up in answer_recovery_map
        """
        # Extract file name without path and extension
        base_filename = xml_path.stem
        
        # Get the pid from the QAPair element
        pid = qa_pair_element.get('pid', '1')
        
        # Try different key formats that might match the CSV
        possible_keys = [
            f"{base_filename}_Sec{pid}.txt",
            f"{base_filename}_Sec{pid}",
            f"{base_filename}_{pid}.txt",
            f"{base_filename}_{pid}",
            f"{base_filename}.txt",
            base_filename
        ]
        
        # Try each possible key format
        for key in possible_keys:
            if key in self.answer_recovery_map:
                return key
        
        return None
    
    def clear_parsed_data(self) -> None:
        """
        Clear the parsed data and reset the counter.
        """
        self.parsed_data = {"unlabeled_qa_examples": []}
        self.question_counter = 0  # Reset counter when clearing data
        self.logger.info("Parsed data cleared and counter reset")
    
    def _get_next_question_id(self) -> str:
        """
        Get the next unique question ID.
        
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
                        question_text = question_element.text.strip() if question_element.text else ""
                        answer_text = answer_element.text.strip() if answer_element is not None and answer_element.text else ""
                        
                        # Check for answer recovery if needed
                        answer_recovered = False
                        if not answer_text and needs_recovery:
                            recovery_key = self._get_recovery_key_from_xml_path(file_path, qa_pair)
                            if recovery_key:
                                recovered_answer = self.answer_recovery_map.get(recovery_key)
                                if recovered_answer:
                                    answer_text = recovered_answer
                                    answer_recovered = True
                                    recovered_count += 1
                        
                        # Extract question type
                        question_type = "unknown"
                        question_type_element = question_element.get('qtype')
                        if question_type_element:
                            question_type = question_type_element
                        
                        # Get risk level
                        risk_level = self._get_risk_level(question_type)
                        
                        # Only add if both question and answer exist
                        if question_text and answer_text:
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
        return {
            "recovery_entries_loaded": len(self.answer_recovery_map),
            "folders_needing_recovery": list(self.folders_needing_recovery),
            "recovery_available": len(self.answer_recovery_map) > 0
        }
    
    def _parse_test_questions_xml(self, xml_path: Path) -> Dict[str, Dict]:
        """
        Parse the TREC-2017-LiveQA-Medical-Test.xml file to extract test questions.
        
        Args:
            xml_path (Path): Path to the test questions XML file
            
        Returns:
            Dict[str, Dict]: Dictionary mapping question IDs to question data
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            questions = {}
            
            for nlm_question in root.findall('NLM-QUESTION'):
                qid = nlm_question.get('qid')
                if not qid:
                    continue
                
                # Get the NIST paraphrase (preferred) or original question
                nist_paraphrase = nlm_question.find('NIST-PARAPHRASE')
                question_text = ""
                if nist_paraphrase is not None and nist_paraphrase.text:
                    question_text = nist_paraphrase.text.strip()
                else:
                    # Fallback to original question
                    original_question = nlm_question.find('Original-Question')
                    if original_question is not None:
                        message = original_question.find('MESSAGE')
                        if message is not None and message.text:
                            question_text = message.text.strip()
                
                # Extract annotations
                annotations = nlm_question.find('ANNOTATIONS')
                focuses = []
                question_type = "unknown"
                
                if annotations is not None:
                    # Get focuses
                    for focus in annotations.findall('FOCUS'):
                        focus_text = focus.text.strip() if focus.text else ""
                        focus_category = focus.get('fcategory', '')
                        if focus_text:
                            focuses.append({
                                "id": focus.get('fid', ''),
                                "text": focus_text,
                                "category": focus_category
                            })
                    
                    # Get question type
                    type_elem = annotations.find('TYPE')
                    if type_elem is not None and type_elem.text:
                        question_type = type_elem.text.strip()
                
                questions[qid] = {
                    "id": qid,
                    "question": question_text,
                    "question_type": question_type,
                    "focuses": focuses
                }
            
            self.logger.info(f"Parsed {len(questions)} test questions from XML")
            return questions
            
        except Exception as e:
            self.logger.error(f"Error parsing test questions XML {xml_path}: {e}")
            return {}
    
    def _parse_qrels_file(self, qrels_path: Path) -> Dict[str, List[Dict]]:
        """
        Parse the qrels file to get question-answer mappings with ratings.
        
        Args:
            qrels_path (Path): Path to the qrels file
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping question IDs to answer ratings
        """
        try:
            qrels_data = {}
            
            with open(qrels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Map numeric question ID to TQ format (1 -> TQ1, 2 -> TQ2, etc.)
                        numeric_qid = parts[0]
                        question_id = f"TQ{numeric_qid}"
                        
                        rating = parts[1]
                        answer_id = parts[2]
                        
                        if question_id not in qrels_data:
                            qrels_data[question_id] = []
                        
                        qrels_data[question_id].append({
                            "answer_id": answer_id,
                            "rating": rating
                        })
            
            self.logger.info(f"Parsed {len(qrels_data)} question-answer mappings from qrels")
            return qrels_data
            
        except Exception as e:
            self.logger.error(f"Error parsing qrels file {qrels_path}: {e}")
            return {}
    
    def _parse_answers_csv(self, csv_path: Path) -> Dict[str, str]:
        """
        Parse the answers CSV file to get answer texts.
        
        Args:
            csv_path (Path): Path to the answers CSV file
            
        Returns:
            Dict[str, str]: Dictionary mapping answer IDs to answer texts
        """
        try:
            answers = {}
            
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    answer_id = row.get('AnswerID', '').strip()
                    answer_text = row.get('Answer', '').strip()
                    
                    if answer_id and answer_text:
                        # Extract just the answer part (after "Answer: " and before closing parentheses)
                        parsed_answer = self._extract_answer_from_recovery_text(answer_text)
                        if parsed_answer:
                            answers[answer_id] = parsed_answer
            
            self.logger.info(f"Parsed {len(answers)} answers from CSV")
            return answers
            
        except Exception as e:
            self.logger.error(f"Error parsing answers CSV {csv_path}: {e}")
            return {}
    
    def generate_test_qna_json(self, output_path: str = "data/processed/test_qna.json") -> None:
        """
        Generate the test QnA JSON file by combining data from test questions XML,
        qrels file, and answers CSV. Includes risk levels for each question.
        
        Args:
            output_path (str): Path where to save the test QnA JSON file
        """
        try:
            # Define file paths
            test_xml_path = Path("data/raw/TestQuestions/TestDataset/TREC-2017-LiveQA-Medical-Test.xml")
            qrels_path = Path("data/raw/MedQuAD/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt")
            answers_csv_path = Path("data/raw/MedQuAD/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-2479-Answers-retrieved-from-MedQuAD.csv")
            
            # Check if files exist
            if not test_xml_path.exists():
                self.logger.error(f"Test questions XML not found: {test_xml_path}")
                return
            
            if not qrels_path.exists():
                self.logger.error(f"Qrels file not found: {qrels_path}")
                return
            
            if not answers_csv_path.exists():
                self.logger.error(f"Answers CSV not found: {answers_csv_path}")
                return
            
            self.logger.info("Starting test QnA generation...")
            
            # Parse all data sources
            questions = self._parse_test_questions_xml(test_xml_path)
            qrels = self._parse_qrels_file(qrels_path)
            answers = self._parse_answers_csv(answers_csv_path)
            
            # Combine the data
            test_qna_data = []
            
            for qid, question_data in questions.items():
                # Get answers and ratings for this question
                question_answers = []
                if qid in qrels:
                    for qrel in qrels[qid]:
                        answer_id = qrel["answer_id"]
                        rating = qrel["rating"]
                        
                        # Get the answer text
                        answer_text = answers.get(answer_id, "")
                        if answer_text:
                            question_answers.append({
                                "answer_id": answer_id,
                                "answer_text": answer_text,
                                "rating": rating
                            })
                
                # Get risk level for this question type
                risk_level = self._get_risk_level(question_data["question_type"])
                
                # Create the combined entry with risk level
                test_entry = {
                    "id": question_data["id"],
                    "question": question_data["question"],
                    "question_type": question_data["question_type"],
                    "risk_level": risk_level,
                    "focuses": question_data["focuses"],
                    "answers": question_answers
                }
                
                test_qna_data.append(test_entry)
            
            # Create output directory if it doesn't exist
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_qna_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Test QnA data saved to: {output_path}")
            self.logger.info(f"Generated {len(test_qna_data)} test question entries")
            
            # Print summary statistics
            total_answers = sum(len(entry["answers"]) for entry in test_qna_data)
            self.logger.info(f"Total answers across all questions: {total_answers}")
            
            # Count by rating
            rating_counts = {}
            for entry in test_qna_data:
                for answer in entry["answers"]:
                    rating = answer["rating"]
                    rating_counts[rating] = rating_counts.get(rating, 0) + 1
            
            self.logger.info(f"Answer rating distribution: {rating_counts}")
            
            # Count by risk level
            risk_counts = {}
            for entry in test_qna_data:
                risk_level = entry["risk_level"]
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            self.logger.info(f"Question risk level distribution: {risk_counts}")
            
        except Exception as e:
            self.logger.error(f"Error generating test QnA JSON: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize the parser with clean logging (set verbose=True for detailed debugging)
    parser = DataParser(verbose=False)
    
    # Generate test QnA JSON file with risk levels
    parser.generate_test_qna_json()
    
    # Get recovery statistics
    # recovery_stats = parser.get_recovery_statistics()
    # print("Answer Recovery Statistics:")
    # print(json.dumps(recovery_stats, indent=2))
    
    # Parse all XML files in the MedQuAD directory
    # parser.parse_directory()
    
    # Or parse specific subdirectories (including those needing recovery)
    # parser.parse_specific_subdirectories(['10_MPlus_ADAM_QA', '11_MPlusDrugs_QA', '12_MPlusHerbsSupplements_QA'])
    
    # Get parsing statistics
    # stats = parser.get_statistics()
    # print("\nParsing Statistics:")
    # print(json.dumps(stats, indent=2))
    
    # Save to SQLite database
    # parser.save_to_database("data/processed/unlabeled_qa.db")