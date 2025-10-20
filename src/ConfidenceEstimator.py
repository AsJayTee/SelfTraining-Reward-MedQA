"""
ConfidenceEstimator.py

Estimates confidence scores for medical answers based on linguistic patterns.
"""

import json
import re
import numpy as np
from pathlib import Path


class ConfidenceEstimator:
    """Estimates confidence scores for medical answers."""
    
    def __init__(self, base_confidence=0.5):
        self.base_confidence = base_confidence
        
        self.uncertainty_patterns = [
            r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bpossibly\b', 
            r'\bperhaps\b', r'\bprobably\b', r'\blikely\b', r'\bunlikely\b',
            r'\bsometimes\b', r'\boften\b', r'\brarely\b', r'\boccasionally\b',
            r'\btypically\b', r'\busually\b', r'\bgenerally\b',
            r'\buncertain\b', r'\bunknown\b', r'\bunclear\b'
        ]
        
        self.overconfidence_patterns = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b', r'\bcertainly\b',
            r'\bclearly\b', r'\bobviously\b', r'\bundoubtedly\b', r'\bwithout\s+doubt\b',
            r'\babsolutely\b', r'\bguaranteed\b', r'\bwill\s+definitely\b',
            r'\bmust\s+be\b', r'\bcan\s+only\s+be\b'
        ]
    
    def estimate_confidence(self, answer_text):
        """Estimate confidence score for an answer."""
        text_lower = answer_text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        uncertainty_count = sum(1 for pattern in self.uncertainty_patterns 
                               if re.search(pattern, text_lower))
        uncertainty_penalty = -min(uncertainty_count * 0.05, 0.20)
        
        overconfidence_count = sum(1 for pattern in self.overconfidence_patterns 
                                  if re.search(pattern, text_lower))
        overconfidence_boost = min(overconfidence_count * 0.03, 0.10)
        
        specificity_count = len(re.findall(r'\d+(\.\d+)?(%| mg| ml| hours| days| weeks| months| years)', text_lower))
        specificity_boost = min(specificity_count * 0.02, 0.08)
        
        medical_terms = ['diagnosis', 'treatment', 'symptoms', 'condition', 'syndrome',
                        'patient', 'clinical', 'medical', 'therapeutic', 'pathology',
                        'disorder', 'disease', 'therapy', 'medication']
        term_count = sum(1 for term in medical_terms if term in text_lower)
        term_density = term_count / word_count if word_count > 0 else 0
        medical_boost = min(term_density * 0.3, 0.05)
        
        final_confidence = (self.base_confidence + 
                           uncertainty_penalty + 
                           overconfidence_boost + 
                           specificity_boost + 
                           medical_boost)
        
        return round(np.clip(final_confidence, 0.05, 0.95), 4)


def main():
    """Process MedQA data and add confidence scores"""
    
    # Fixed: Use forward slashes and Path for cross-platform compatibility
    input_file = Path("data/processed/test_qna.json")
    output_file = Path("data/processed/test_qna_with_confidence.json")
    
    print("Processing MedQA data...")
    
    # Load data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Process data
    estimator = ConfidenceEstimator()
    
    for item in data:
        for answer in item['answers']:
            confidence = estimator.estimate_confidence(answer['answer_text'])
            answer['confidence_score'] = confidence
    
    # Calculate stats
    total_answers = sum(len(item['answers']) for item in data)
    all_scores = [answer['confidence_score'] for item in data for answer in item['answers']]
    
    # Save data
    try:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {total_answers} answers to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")
        return
    
    # Show results
    print(f"Confidence range: {min(all_scores):.4f} to {max(all_scores):.4f}")
    print(f"Mean confidence: {np.mean(all_scores):.4f}")


if __name__ == "__main__":
    main()