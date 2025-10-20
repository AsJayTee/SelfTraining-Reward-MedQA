"""
PreferencePairCreator.py

A class to create preference pairs from MedQuAD test data with confidence scores.
Implements score-primary preference creation with risk-aware confidence tie-breaking
and confidence penalty mechanism for high-risk medical scenarios.
"""

import json
import random
import logging
from pathlib import Path
from itertools import combinations
from typing import List, Dict
import numpy as np


class PreferencePairCreator:
    """
    Creates preference pairs from medical Q&A data with risk-aware confidence handling.
    
    Key Features:
    - Score-primary preference creation (score difference >= min_score_diff)
    - Risk-aware confidence tie-breaking (high-risk prefers lower confidence)
    - Confidence penalty mechanism for dangerous high-confidence + low-score answers
    - Configurable thresholds and comprehensive metadata tracking
    """
    
    def __init__(self, min_score_diff: float = 1.0, random_seed: int = 42):
        """
        Initialize the PreferencePairCreator.
        
        Args:
            min_score_diff (float): Minimum score difference for clear preference (default: 1.0)
            random_seed (int): Random seed for reproducible tie-breaking (default: 42)
        """
        self.min_score_diff = min_score_diff
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = {
            'total_questions': 0,
            'total_answers': 0,
            'total_pairs': 0,
            'score_based_pairs': 0,
            'confidence_tiebreak_pairs': 0,
            'pairs_with_penalties': 0,
            'total_penalty_amount': 0.0,
            'risk_level_distribution': {},
            'score_difference_distribution': {}
        }
    
    def parse_rating(self, rating: str) -> int:
        """
        Extract numeric score from rating string (e.g., "3-Good" -> 3).
        
        Args:
            rating (str): Rating string in format "X-Description"
            
        Returns:
            int: Numeric score (1-4)
        """
        try:
            match = rating.split('-')[0]
            return int(match)
        except (ValueError, IndexError):
            self.logger.warning(f"Could not parse rating: {rating}, defaulting to 1")
            return 1
    
    def confidence_tiebreaker(self, ans_a: Dict, ans_b: Dict, risk_level: str) -> Dict[str, Dict]:
        """
        Risk-aware tie-breaking when scores are equal or very close.
        
        For high-risk questions: Prefer LOWER confidence (medical caution)
        For low/medium-risk: Prefer HIGHER confidence (shows knowledge)
        
        Args:
            ans_a, ans_b: Two answers with equal/similar scores
            risk_level: Risk level of the question
            
        Returns:
            Dict with 'preferred' and 'rejected' answers
        """
        conf_a = ans_a['confidence_score']
        conf_b = ans_b['confidence_score']
        
        if risk_level == 'high':
            # High risk: prefer lower confidence (medical caution)
            if conf_a < conf_b:
                return {'preferred': ans_a, 'rejected': ans_b}
            elif conf_b < conf_a:
                return {'preferred': ans_b, 'rejected': ans_a}
        else:
            # Low/Medium risk: prefer higher confidence
            if conf_a > conf_b:
                return {'preferred': ans_a, 'rejected': ans_b}
            elif conf_b > conf_a:
                return {'preferred': ans_b, 'rejected': ans_a}
        
        # If confidences are equal, random choice
        if random.random() < 0.5:
            return {'preferred': ans_a, 'rejected': ans_b}
        else:
            return {'preferred': ans_b, 'rejected': ans_a}
    
    def calculate_confidence_penalty(self, preferred: Dict, rejected: Dict, risk_level: str) -> float:
        """
        Calculate confidence penalty for dangerous high-confidence + low-score answers.
        
        Penalty applied when:
        - High-risk question
        - Preferred answer has lower score but HIGHER confidence
        - This combination is particularly dangerous in medical contexts
        
        Args:
            preferred: Preferred answer dict
            rejected: Rejected answer dict
            risk_level: Risk level of the question
            
        Returns:
            float: Confidence penalty (0.0 if no penalty applied)
        """
        if risk_level != 'high':
            return 0.0
        
        # Check for dangerous pattern: lower score but higher confidence
        if (preferred['score'] < rejected['score'] and 
            preferred['confidence_score'] > rejected['confidence_score']):
            
            # Calculate penalty magnitude
            score_gap = rejected['score'] - preferred['score']
            conf_gap = preferred['confidence_score'] - rejected['confidence_score']
            
            # Penalty scales with both gaps
            penalty = score_gap * conf_gap * 0.5
            return round(penalty, 4)
        
        return 0.0
    
    def create_preference_pairs(self, question: Dict, answers: List[Dict], risk_level: str) -> List[Dict]:
        """
        Create preference pairs from a set of answers for a single question.
        
        Args:
            question: Question dictionary
            answers: List of answer dictionaries with ratings and confidence scores
            risk_level: Risk level classification
            
        Returns:
            List of preference pair dictionaries
        """
        pairs = []
        
        # Parse scores for all answers
        for answer in answers:
            answer['score'] = self.parse_rating(answer['rating'])
        
        # Generate all possible pairs
        for ans_a, ans_b in combinations(answers, 2):
            score_a = ans_a['score']
            score_b = ans_b['score']
            score_diff = abs(score_a - score_b)
            
            # Skip if scores are too similar for clear preference
            if score_diff < self.min_score_diff:
                continue
            
            # Determine preferred and rejected based on score
            if score_a > score_b:
                preferred = ans_a
                rejected = ans_b
                creation_reason = 'score_diff'
                confidence_penalty = self.calculate_confidence_penalty(preferred, rejected, risk_level)
            elif score_b > score_a:
                preferred = ans_b
                rejected = ans_a
                creation_reason = 'score_diff'
                confidence_penalty = self.calculate_confidence_penalty(preferred, rejected, risk_level)
                
            else:
                # Tie-breaking using confidence with risk-aware strategy
                tiebreak_result = self.confidence_tiebreaker(ans_a, ans_b, risk_level)
                preferred = tiebreak_result['preferred']
                rejected = tiebreak_result['rejected']
                creation_reason = 'confidence_tiebreak'
                confidence_penalty = 0.0  # No additional penalty for ties
            
            # Create preference pair
            pair = {
                'question_id': question['id'],
                'question_text': question['question'],
                'question_type': question['question_type'],
                'risk_level': risk_level,
                'preferred_answer': {
                    'answer_id': preferred['answer_id'],
                    'answer_text': preferred['answer_text'],
                    'rating': preferred['rating'],
                    'score': preferred['score'],
                    'confidence_score': preferred['confidence_score']
                },
                'rejected_answer': {
                    'answer_id': rejected['answer_id'],
                    'answer_text': rejected['answer_text'],
                    'rating': rejected['rating'], 
                    'score': rejected['score'],
                    'confidence_score': rejected['confidence_score']
                },
                'score_difference': score_diff,
                'confidence_penalty': confidence_penalty,
                'creation_reason': creation_reason
            }
            
            pairs.append(pair)
            
            # Update statistics
            self.stats['total_pairs'] += 1
            if creation_reason == 'score_diff':
                self.stats['score_based_pairs'] += 1
            else:
                self.stats['confidence_tiebreak_pairs'] += 1
                
            if confidence_penalty > 0:
                self.stats['pairs_with_penalties'] += 1
                self.stats['total_penalty_amount'] += confidence_penalty
                
            # Track score difference distribution
            score_diff_bucket = f"{int(score_diff)}"
            self.stats['score_difference_distribution'][score_diff_bucket] = \
                self.stats['score_difference_distribution'].get(score_diff_bucket, 0) + 1
        
        return pairs
    
    def process_dataset(self, data: List[Dict]) -> List[Dict]:
        """
        Process entire dataset to create preference pairs.
        
        Args:
            data (List[Dict]): List of questions with answers and confidence scores
            
        Returns:
            List[Dict]: List of all preference pairs
        """
        all_pairs = []
        
        for question_data in data:
            self.stats['total_questions'] += 1
            self.stats['total_answers'] += len(question_data['answers'])
            
            risk_level = question_data['risk_level']
            self.stats['risk_level_distribution'][risk_level] = \
                self.stats['risk_level_distribution'].get(risk_level, 0) + 1
            
            pairs = self.create_preference_pairs(
                question_data, 
                question_data['answers'], 
                risk_level
            )
            
            all_pairs.extend(pairs)
            
            if self.stats['total_questions'] % 50 == 0:
                self.logger.info(f"Processed {self.stats['total_questions']} questions, "
                               f"generated {len(all_pairs)} pairs so far...")
        
        return all_pairs
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about preference pair creation."""
        stats_copy = self.stats.copy()
        
        # Calculate derived statistics
        if stats_copy['total_pairs'] > 0:
            stats_copy['score_based_percentage'] = \
                (stats_copy['score_based_pairs'] / stats_copy['total_pairs']) * 100
            stats_copy['tiebreak_percentage'] = \
                (stats_copy['confidence_tiebreak_pairs'] / stats_copy['total_pairs']) * 100
            stats_copy['penalty_percentage'] = \
                (stats_copy['pairs_with_penalties'] / stats_copy['total_pairs']) * 100
            stats_copy['average_penalty'] = \
                stats_copy['total_penalty_amount'] / max(1, stats_copy['pairs_with_penalties'])
        
        return stats_copy
    
    def save_pairs_to_file(self, pairs: List[Dict], output_path: str) -> None:
        """
        Save preference pairs to JSON file.
        
        Args:
            pairs (List[Dict]): Preference pairs to save
            output_path (str): Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(pairs)} preference pairs to {output_path}")


def main():
    """
    Main execution function to process MedQuAD data and create preference pairs.
    """
    # Configuration - Fixed: Use forward slashes and Path for cross-platform compatibility
    input_file = Path("data/processed/test_qna_with_confidence.json")
    output_file = Path("data/processed/preference_pairs.json")
    stats_file = Path("data/processed/preference_pair_stats.json")
    min_score_diff = 1.0  # Configurable threshold
    
    print("=== MedQuAD Preference Pair Creator ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Minimum score difference: {min_score_diff}")
    
    # Initialize creator
    creator = PreferencePairCreator(min_score_diff=min_score_diff)
    
    try:
        # Load data
        print(f"\nLoading data from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions")
        
        # Process data to create preference pairs
        print("\nProcessing questions to create preference pairs...")
        preference_pairs = creator.process_dataset(data)
        
        # Save preference pairs
        print(f"\nSaving {len(preference_pairs)} preference pairs...")
        creator.save_pairs_to_file(preference_pairs, output_file)
        
        # Get and display statistics
        stats = creator.get_statistics()
        print(f"\n=== PROCESSING STATISTICS ===")
        print(f"Total questions processed: {stats['total_questions']}")
        print(f"Total answers processed: {stats['total_answers']}")
        print(f"Total preference pairs created: {stats['total_pairs']}")
        print(f"Score-based pairs: {stats['score_based_pairs']} ({stats.get('score_based_percentage', 0):.1f}%)")
        print(f"Confidence tie-break pairs: {stats['confidence_tiebreak_pairs']} ({stats.get('tiebreak_percentage', 0):.1f}%)")
        print(f"Pairs with confidence penalties: {stats['pairs_with_penalties']} ({stats.get('penalty_percentage', 0):.1f}%)")
        
        if stats['pairs_with_penalties'] > 0:
            print(f"Total penalty amount: {stats['total_penalty_amount']:.4f}")
            print(f"Average penalty per penalized pair: {stats.get('average_penalty', 0):.4f}")
        
        print(f"\nRisk level distribution:")
        for risk_level, count in stats['risk_level_distribution'].items():
            percentage = (count / stats['total_questions']) * 100
            print(f"  {risk_level}: {count} questions ({percentage:.1f}%)")
        
        print(f"\nScore difference distribution:")
        for score_diff, count in sorted(stats['score_difference_distribution'].items()):
            percentage = (count / stats['total_pairs']) * 100
            print(f"  Score diff {score_diff}: {count} pairs ({percentage:.1f}%)")
        
        # Save statistics
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to {stats_file}")
        
        # Sample output
        print(f"\n=== SAMPLE PREFERENCE PAIRS ===")
        
        # Show a high-penalty pair if available
        high_penalty_pairs = [p for p in preference_pairs if p['confidence_penalty'] > 0.1]
        if high_penalty_pairs:
            sample_pair = high_penalty_pairs[0]
            print(f"\nHigh-penalty pair example:")
            print(f"Question: {sample_pair['question_text']}")
            print(f"Risk Level: {sample_pair['risk_level']}")
            print(f"Preferred (Score {sample_pair['preferred_answer']['score']}, "
                  f"Conf {sample_pair['preferred_answer']['confidence_score']:.3f}): "
                  f"{sample_pair['preferred_answer']['answer_text'][:100]}...")
            print(f"Rejected (Score {sample_pair['rejected_answer']['score']}, "
                  f"Conf {sample_pair['rejected_answer']['confidence_score']:.3f}): "
                  f"{sample_pair['rejected_answer']['answer_text'][:100]}...")
            print(f"Confidence Penalty: {sample_pair['confidence_penalty']:.4f}")
        
        # Show a regular pair
        regular_pairs = [p for p in preference_pairs if p['creation_reason'] == 'score_diff' and p['confidence_penalty'] == 0]
        if regular_pairs:
            sample_pair = regular_pairs[0]
            print(f"\nRegular score-based pair example:")
            print(f"Question: {sample_pair['question_text']}")
            print(f"Score Difference: {sample_pair['score_difference']}")
            print(f"Preferred (Score {sample_pair['preferred_answer']['score']}): "
                  f"{sample_pair['preferred_answer']['answer_text'][:100]}...")
            print(f"Rejected (Score {sample_pair['rejected_answer']['score']}): "
                  f"{sample_pair['rejected_answer']['answer_text'][:100]}...")
        
        print(f"\nPreference pair creation completed successfully!")
        print(f"Output files:")
        print(f" - Preference pairs: {output_file}")
        print(f" - Statistics: {stats_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        print("Please ensure you have run ConfidenceEstimator.py to generate the input file.")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()