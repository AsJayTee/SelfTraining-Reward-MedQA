"""
pipeline.py

Main orchestration pipeline for self-learning data generation.
Loads unlabeled Q&A pairs, generates confidence-calibrated preference pairs,
and saves outputs for training.

Author: Jin Thau - DSA4213 Group 18
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

from .ResponseGenerator import ResponseGenerator


class SelfLearningPipeline:
    """
    Orchestrates the self-learning data generation process.
    
    Pipeline Flow:
    1. Load unlabeled Q&A from database
    2. Generate better/worse versions using LLM
    3. Create preference pairs with confidence scores
    4. Save outputs for training
    """
    
    def __init__(self, 
                 db_path: str = "data/processed/unlabeled_qa.db",
                 output_dir: str = "data/processed/self_learning",
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device: str = "cuda"):
        """
        Initialize the pipeline.
        
        Args:
            db_path: Path to unlabeled Q&A database
            output_dir: Directory to save outputs
            model_name: HuggingFace model for generation
            device: Device to run on (cuda/cpu)
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Pipeline initialized")
        self.logger.info(f"  Database: {self.db_path}")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  Model: {self.model_name}")
        self.logger.info(f"  Device: {self.device}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "pipeline.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_unlabeled_data(self, 
                           limit: Optional[int] = None,
                           offset: int = 0,
                           risk_level_filter: Optional[str] = None) -> List[Dict]:
        """
        Load unlabeled Q&A examples from SQLite database.
        
        Args:
            limit: Maximum number of examples to load (None = all)
            offset: Number of examples to skip (for pagination)
            risk_level_filter: Filter by risk level (e.g., "High Risk")
        
        Returns:
            List of unlabeled Q&A pairs
        
        Raises:
            FileNotFoundError: If database doesn't exist
            sqlite3.Error: If database query fails
        """
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                f"Please run DataParser.py first to generate the database."
            )
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT id as question_id, question as question_text, question_type, 
                    answer as answer_text, id as answer_id, medical_focus, risk_level
                FROM unlabeled_qa
            """
            
            # Add filter if specified
            params = []
            if risk_level_filter:
                query += " WHERE risk_level = ?"
                params.append(risk_level_filter)
            
            # Add ordering for consistency
            query += " ORDER BY question_id"
            
            # Add pagination
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            self.logger.info(f"Loading data from database...")
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dicts
            qa_pairs = []
            for row in rows:
                qa_pairs.append({
                    "question_id": row[0],
                    "question_text": row[1],
                    "question_type": row[2],
                    "answer_text": row[3],
                    "answer_id": row[4],
                    "medical_focus": row[5],
                    "risk_level": row[6]
                })
            
            self.logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
            
            # Log distribution
            risk_counts = {}
            for qa in qa_pairs:
                risk = qa['risk_level']
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            self.logger.info(f"Risk level distribution:")
            for risk, count in sorted(risk_counts.items()):
                self.logger.info(f"  {risk}: {count}")
            
            return qa_pairs
        
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            raise
    
    def save_checkpoint(self, data: List[Dict], checkpoint_name: str):
        """
        Save checkpoint data to JSON file.
        
        Args:
            data: Data to save
            checkpoint_name: Name of checkpoint file (without extension)
        """
        checkpoint_file = self.output_dir / f"{checkpoint_name}.json"
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_file}")
            self.logger.info(f"  Size: {len(data)} items")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def run_iteration(self, 
                     iteration: int,
                     num_examples: int = 15000,
                     offset: int = 0,
                     test_mode: bool = False) -> str:
        """
        Run one iteration of self-learning data generation.
        
        Args:
            iteration: Iteration number (1, 2, 3)
            num_examples: Number of unlabeled examples to process
            offset: Offset for pagination (use different data each iteration)
            test_mode: If True, process only 10 examples for testing
        
        Returns:
            Path to output file
        """
        start_time = time.time()
        
        if test_mode:
            num_examples = 10
            self.logger.warning("TEST MODE: Processing only 10 examples")
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"ITERATION {iteration}: Self-Learning Data Generation")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Processing {num_examples} examples (offset: {offset})")
        
        try:
            # Step 1: Load unlabeled data
            self.logger.info(f"\n--- Step 1: Loading Unlabeled Data ---")
            qa_pairs = self.load_unlabeled_data(
                limit=num_examples,
                offset=offset
            )
            
            if not qa_pairs:
                raise ValueError("No data loaded from database!")
            
            # Save checkpoint
            checkpoint_name = f"iter{iteration}_01_unlabeled"
            self.save_checkpoint(qa_pairs, checkpoint_name)
            
            # Step 2: Generate preference pairs
            self.logger.info(f"\n--- Step 2: Generating Preference Pairs ---")
            self.logger.info(f"Loading model: {self.model_name}")
            
            generator = ResponseGenerator(
                model_name=self.model_name,
                device=self.device
            )
            
            self.logger.info(f"Generating better/worse versions...")
            preference_pairs = generator.generate_batch(
                qa_pairs,
                save_every=50
            )
            
            # Cleanup model from memory
            self.logger.info("Cleaning up model from GPU memory...")
            generator.cleanup()
            
            if not preference_pairs:
                raise ValueError("No preference pairs generated!")
            
            # Save checkpoint
            checkpoint_name = f"iter{iteration}_02_all_pairs"
            self.save_checkpoint(preference_pairs, checkpoint_name)
            
            # Step 3: Save final output
            self.logger.info(f"\n--- Step 3: Saving Final Output ---")
            output_file = self.output_dir / f"iter{iteration}_synthetic_pairs.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(preference_pairs, f, indent=2, ensure_ascii=False)
            
            # Calculate statistics
            end_time = time.time()
            duration = end_time - start_time
            
            stats = self._calculate_statistics(preference_pairs, qa_pairs, duration)
            
            # Save statistics
            stats_file = self.output_dir / f"iter{iteration}_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            # Print summary
            self._print_summary(iteration, output_file, stats)
            
            return str(output_file)
        
        except Exception as e:
            self.logger.error(f"Error in iteration {iteration}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _calculate_statistics(self, 
                             preference_pairs: List[Dict],
                             qa_pairs: List[Dict],
                             duration: float) -> Dict:
        """Calculate comprehensive statistics."""
        import numpy as np
        
        stats = {
            "iteration_info": {
                "total_qa_pairs": len(qa_pairs),
                "total_preference_pairs": len(preference_pairs),
                "expansion_ratio": len(preference_pairs) / len(qa_pairs) if qa_pairs else 0,
                "duration_seconds": round(duration, 2),
                "duration_minutes": round(duration / 60, 2),
                "pairs_per_second": round(len(preference_pairs) / duration, 2) if duration > 0 else 0
            },
            "risk_distribution": {},
            "creation_reasons": {},
            "confidence_penalties": {
                "count": 0,
                "total": 0.0,
                "mean": 0.0,
                "max": 0.0,
                "by_risk_level": {}
            },
            "teaching_objectives": {},
            "confidence_scores": {
                "preferred": [],
                "rejected": []
            }
        }
        
        penalties = []
        
        for pair in preference_pairs:
            # Risk distribution
            risk = pair.get("risk_level", "Unknown")
            stats["risk_distribution"][risk] = stats["risk_distribution"].get(risk, 0) + 1
            
            # Creation reasons
            reason = pair.get("creation_reason", "unknown")
            stats["creation_reasons"][reason] = stats["creation_reasons"].get(reason, 0) + 1
            
            # Teaching objectives
            teaches = pair.get("teaches", "unknown")
            stats["teaching_objectives"][teaches] = stats["teaching_objectives"].get(teaches, 0) + 1
            
            # Confidence penalties
            penalty = pair.get("confidence_penalty", 0.0)
            if penalty > 0:
                penalties.append(penalty)
                stats["confidence_penalties"]["count"] += 1
                stats["confidence_penalties"]["total"] += penalty
                
                # Track by risk level
                if risk not in stats["confidence_penalties"]["by_risk_level"]:
                    stats["confidence_penalties"]["by_risk_level"][risk] = []
                stats["confidence_penalties"]["by_risk_level"][risk].append(penalty)
            
            # Confidence scores
            if "estimated_confidence" in pair.get("preferred_answer", {}):
                stats["confidence_scores"]["preferred"].append(
                    pair["preferred_answer"]["estimated_confidence"]
                )
            if "estimated_confidence" in pair.get("rejected_answer", {}):
                stats["confidence_scores"]["rejected"].append(
                    pair["rejected_answer"]["estimated_confidence"]
                )
        
        # Calculate penalty statistics
        if penalties:
            stats["confidence_penalties"]["mean"] = round(np.mean(penalties), 4)
            stats["confidence_penalties"]["max"] = round(max(penalties), 4)
            
            # Aggregate by risk level
            for risk, risk_penalties in stats["confidence_penalties"]["by_risk_level"].items():
                stats["confidence_penalties"]["by_risk_level"][risk] = {
                    "count": len(risk_penalties),
                    "mean": round(np.mean(risk_penalties), 4),
                    "max": round(max(risk_penalties), 4)
                }
        
        # Calculate confidence score statistics
        for key in ["preferred", "rejected"]:
            if stats["confidence_scores"][key]:
                scores = stats["confidence_scores"][key]
                stats["confidence_scores"][key] = {
                    "mean": round(np.mean(scores), 4),
                    "std": round(np.std(scores), 4),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4)
                }
            else:
                stats["confidence_scores"][key] = None
        
        return stats
    
    def _print_summary(self, iteration: int, output_file: Path, stats: Dict):
        """Print iteration summary."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"ITERATION {iteration} COMPLETE")
        self.logger.info(f"{'='*70}")
        
        # Basic info
        info = stats["iteration_info"]
        self.logger.info(f"\nGeneration Summary:")
        self.logger.info(f"  Input Q&A pairs: {info['total_qa_pairs']}")
        self.logger.info(f"  Output preference pairs: {info['total_preference_pairs']}")
        self.logger.info(f"  Expansion ratio: {info['expansion_ratio']:.2f}x")
        self.logger.info(f"  Duration: {info['duration_minutes']:.2f} minutes")
        self.logger.info(f"  Speed: {info['pairs_per_second']:.2f} pairs/second")
        
        # Risk distribution
        self.logger.info(f"\nRisk Level Distribution:")
        for risk, count in sorted(stats["risk_distribution"].items()):
            pct = (count / info['total_preference_pairs']) * 100
            self.logger.info(f"  {risk}: {count} ({pct:.1f}%)")
        
        # Teaching objectives
        self.logger.info(f"\nTeaching Objectives:")
        for objective, count in sorted(stats["teaching_objectives"].items()):
            pct = (count / info['total_preference_pairs']) * 100
            self.logger.info(f"  {objective}: {count} ({pct:.1f}%)")
        
        # Confidence penalties
        penalties = stats["confidence_penalties"]
        if penalties["count"] > 0:
            self.logger.info(f"\nConfidence Penalties (High-Risk Overconfidence):")
            self.logger.info(f"  Total pairs with penalty: {penalties['count']}")
            self.logger.info(f"  Mean penalty: {penalties['mean']:.4f}")
            self.logger.info(f"  Max penalty: {penalties['max']:.4f}")
            self.logger.info(f"  Total penalty amount: {penalties['total']:.4f}")
            
            self.logger.info(f"\n  By Risk Level:")
            for risk, risk_stats in penalties["by_risk_level"].items():
                self.logger.info(f"    {risk}: {risk_stats['count']} pairs, "
                               f"mean={risk_stats['mean']:.4f}, "
                               f"max={risk_stats['max']:.4f}")
        
        # Confidence scores
        self.logger.info(f"\nConfidence Score Distribution:")
        for answer_type in ["preferred", "rejected"]:
            conf_stats = stats["confidence_scores"][answer_type]
            if conf_stats:
                self.logger.info(f"  {answer_type.capitalize()}: "
                               f"mean={conf_stats['mean']:.4f}, "
                               f"std={conf_stats['std']:.4f}, "
                               f"range=[{conf_stats['min']:.4f}, {conf_stats['max']:.4f}]")
        
        # Output files
        self.logger.info(f"\nOutput Files:")
        self.logger.info(f"  Main output: {output_file}")
        self.logger.info(f"  Statistics: {output_file.parent / f'iter{iteration}_statistics.json'}")
        self.logger.info(f"  Log file: {output_file.parent / 'pipeline.log'}")
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"✓ Ready to hand off to Crescent for training!")
        self.logger.info(f"{'='*70}\n")
    
    def run_all_iterations(self, 
                          num_examples_per_iter: int = 15000,
                          num_iterations: int = 3,
                          test_mode: bool = False):
        """
        Run all iterations sequentially.
        
        Args:
            num_examples_per_iter: Examples per iteration
            num_iterations: Number of iterations to run
            test_mode: If True, run in test mode
        """
        self.logger.info(f"\n{'#'*70}")
        self.logger.info(f"STARTING SELF-LEARNING PIPELINE")
        self.logger.info(f"{'#'*70}")
        self.logger.info(f"Total iterations: {num_iterations}")
        self.logger.info(f"Examples per iteration: {num_examples_per_iter}")
        
        output_files = []
        
        for i in range(1, num_iterations + 1):
            # Use different data for each iteration
            offset = (i - 1) * num_examples_per_iter
            
            output_file = self.run_iteration(
                iteration=i,
                num_examples=num_examples_per_iter,
                offset=offset,
                test_mode=test_mode
            )
            
            output_files.append(output_file)
            
            # Short pause between iterations
            if i < num_iterations:
                self.logger.info(f"\nPausing 30 seconds before next iteration...")
                time.sleep(30)
        
        # Final summary
        self.logger.info(f"\n{'#'*70}")
        self.logger.info(f"ALL ITERATIONS COMPLETE")
        self.logger.info(f"{'#'*70}")
        self.logger.info(f"\nGenerated {num_iterations} synthetic datasets:")
        for i, output_file in enumerate(output_files, 1):
            self.logger.info(f"  Iteration {i}: {output_file}")
        
        self.logger.info(f"\n✓ All data ready for Crescent's training pipeline!")


# Main execution
if __name__ == "__main__":
    """
    Example usage:
    
    # Test mode (10 examples)
    python -m src.self_learning.pipeline --test
    
    # Single iteration
    python -m src.self_learning.pipeline --iteration 1 --num-examples 15000
    
    # All iterations
    python -m src.self_learning.pipeline --all
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Learning Data Generation Pipeline")
    parser.add_argument("--iteration", type=int, help="Run specific iteration (1, 2, or 3)")
    parser.add_argument("--num-examples", type=int, default=15000, help="Number of examples per iteration")
    parser.add_argument("--all", action="store_true", help="Run all 3 iterations")
    parser.add_argument("--test", action="store_true", help="Test mode (10 examples only)")
    parser.add_argument("--db-path", type=str, default="data/processed/unlabeled_qa.db", help="Path to database")
    parser.add_argument("--output-dir", type=str, default="data/processed/self_learning", help="Output directory")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SelfLearningPipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device 
    )
    
    # Run based on arguments
    if args.test:
        print("Running in TEST MODE (10 examples)")
        pipeline.run_iteration(iteration=1, num_examples=10, test_mode=True)
    
    elif args.all:
        print(f"Running all iterations ({args.num_examples} examples each)")
        pipeline.run_all_iterations(
            num_examples_per_iter=args.num_examples,
            num_iterations=3,
            test_mode=False
        )
    
    elif args.iteration:
        print(f"Running iteration {args.iteration}")
        offset = (args.iteration - 1) * args.num_examples
        pipeline.run_iteration(
            iteration=args.iteration,
            num_examples=args.num_examples,
            offset=offset,
            test_mode=False
        )
    
    else:
        print("Please specify --iteration, --all, or --test")
        parser.print_help()