import json
import numpy as np
import re

class ConfidenceEstimator:
    """Estimates confidence level from medical answer text"""
    
    def __init__(self):
        self.base_confidence = 0.75
        
        # Uncertainty keywords (reduce confidence)
        self.uncertainty_keywords = {
            'strong': ['unknown', 'unclear', 'uncertain', 'unsure', 'not known',
                      'insufficient evidence', 'conflicting evidence', 'controversial',
                      'not well understood', 'poorly understood', 'inconclusive',
                      'more research needed', 'difficult to determine'],
            'moderate': ['may', 'might', 'could', 'possibly', 'perhaps', 'probably',
                        'likely', 'unlikely', 'generally', 'usually', 'typically',
                        'often', 'sometimes', 'tends to', 'appears to', 'seems to',
                        'suggests', 'indicates', 'can vary', 'may vary'],
            'mild': ['consult your doctor', 'seek medical advice', 'discuss with',
                    'individual results may vary', 'case by case', 'depends on']
        }
        
        # Overconfident keywords (increase confidence)
        self.overconfident_keywords = {
            'extreme': ['always', 'never', 'definitely', 'certainly', 'absolutely',
                       'without question', 'undoubtedly', 'guaranteed', 'will always',
                       'will never', 'impossible', 'no exceptions', 'in all cases'],
            'strong': ['clearly', 'obviously', 'evidently', 'without doubt',
                      'proven fact', 'established fact', 'well-known fact',
                      'science shows', 'research proves'],
            'moderate': ['proven', 'established', 'confirmed', 'well-documented',
                        'standard', 'recognized', 'shows', 'demonstrates']
        }
        
        self.adjustments = {
            'uncertainty': {'strong': -0.25, 'moderate': -0.12, 'mild': -0.08},
            'overconfident': {'extreme': 0.20, 'strong': 0.15, 'moderate': 0.08}
        }
    
    def estimate_confidence(self, text: str) -> float:
        """Estimate confidence score for text"""
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Length normalization factor (reference: 50 words)
        length_factor = min(50 / max(word_count, 10), 1.0)
        
        # Calculate uncertainty penalty (with length normalization)
        uncertainty_penalty = 0.0
        for level, keywords in self.uncertainty_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    uncertainty_penalty += self.adjustments['uncertainty'][level]
        
        uncertainty_penalty *= length_factor
        
        # Calculate overconfidence boost (no normalization)
        overconfidence_boost = 0.0
        for level, keywords in self.overconfident_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    overconfidence_boost += self.adjustments['overconfident'][level]
        
        # Text features
        specificity_count = len(re.findall(r'\b\d+%|\b\d+\.\d+|\b\d+\s?(mg|ml|hours|days|weeks|months|years)', text_lower))
        specificity_boost = min(specificity_count * 0.02, 0.08)
        
        medical_terms = ['diagnosis', 'treatment', 'symptoms', 'condition', 'syndrome',
                        'patient', 'clinical', 'medical', 'therapeutic', 'pathology',
                        'disorder', 'disease', 'therapy', 'medication']
        term_count = sum(1 for term in medical_terms if term in text_lower)
        term_density = term_count / word_count if word_count > 0 else 0
        medical_boost = min(term_density * 0.3, 0.05)
        
        # Final score
        final_confidence = (self.base_confidence + 
                           uncertainty_penalty + 
                           overconfidence_boost + 
                           specificity_boost + 
                           medical_boost)
        
        return round(np.clip(final_confidence, 0.05, 0.95), 4)

def main():
    """Process MedQA data and add confidence scores"""
    
    input_file = r"data\processed\test_qna.json"
    output_file = r"data\processed\test_qna_with_confidence.json"
    
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