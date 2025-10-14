"""
Generate better and worse versions aligned with ConfidenceEstimator.py.
Uses the exact same keywords and measurement approach.
"""
import json
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import random

class ResponseGenerator:
    """Generate confidence-calibrated answers aligned with ConfidenceEstimator."""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device: str = "cuda",
                 max_memory: str = "13GB"):
        """Initialize with exact ConfidenceEstimator keyword lists."""
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: max_memory}
        )
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        # EXACT keywords from ConfidenceEstimator.py
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
        
        # Confidence score effects (from ConfidenceEstimator)
        self.adjustments = {
            'uncertainty': {'strong': -0.25, 'moderate': -0.12, 'mild': -0.08},
            'overconfident': {'extreme': 0.20, 'strong': 0.15, 'moderate': 0.08}
        }
        
        # Base confidence from ConfidenceEstimator
        self.base_confidence = 0.75
    
    def _generate(self, prompt: str, max_new_tokens: int = 256) -> Optional[str]:
        """Helper method to generate text from prompt."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            if '\n\n' in generated_text:
                generated_text = generated_text.split('\n\n')[0]
            
            return generated_text if len(generated_text) > 10 else None
        
        except Exception as e:
            print(f"Error generating: {e}")
            return None
    
    def generate_better_answer(self, 
                               question: str,
                               original_answer: str,
                               question_type: str,
                               risk_level: str) -> Optional[str]:
        """
        Generate well-calibrated answer using APPROPRIATE uncertainty language.
        
        Target confidence: 0.65-0.75 (safe range)
        Uses mild-moderate uncertainty keywords from ConfidenceEstimator.
        """
        # Select appropriate keywords based on risk level
        if risk_level == "High Risk":
            # Use strong uncertainty + mild qualifiers
            example_phrases = random.sample(self.uncertainty_keywords['mild'], 2)
            uncertainty_guidance = f"Use cautious language like: '{example_phrases[0]}', '{example_phrases[1]}'"
            
        elif risk_level == "Medium Risk":
            # Use moderate uncertainty
            example_phrases = random.sample(self.uncertainty_keywords['moderate'], 2)
            uncertainty_guidance = f"Use appropriate qualifiers like: '{example_phrases[0]}', '{example_phrases[1]}'"
            
        else:  # Low Risk
            # Can use fewer qualifiers but still some
            example_phrases = random.sample(self.uncertainty_keywords['moderate'], 1)
            uncertainty_guidance = f"Be informative but include qualifiers like: '{example_phrases[0]}'"
        
        prompt = f"""You are improving a medical answer with APPROPRIATE confidence calibration.

Question: {question}
Question Type: {question_type}
Risk Level: {risk_level}

Original Answer:
{original_answer}

Task: Create an IMPROVED version that:
- Is more accurate and complete
- {uncertainty_guidance}
- AVOID absolute language: 'always', 'never', 'definitely', 'certainly', 'guaranteed'
- Include appropriate medical caveats
- 2-4 sentences

Improved Answer:"""

        return self._generate(prompt, max_new_tokens=350)
    
    def generate_overconfident_answer(self,
                                     question: str,
                                     original_answer: str,
                                     question_type: str,
                                     risk_level: str) -> Optional[str]:
        """
        Generate OVERCONFIDENT answer using extreme keywords.
        
        Target confidence: 0.85-0.95 (dangerous overconfidence)
        Uses extreme + strong overconfident keywords from ConfidenceEstimator.
        
        This is what we want to PENALIZE, especially for high-risk.
        """
        # Select overconfident keywords to inject
        extreme_keywords = random.sample(self.overconfident_keywords['extreme'], 2)
        strong_keywords = random.sample(self.overconfident_keywords['strong'], 1)
        
        if risk_level == "High Risk":
            danger_note = f"""
CRITICAL: This is a HIGH-RISK medical question. Your task is to create a DANGEROUSLY overconfident answer (for training purposes).
This represents the type of response we want to AVOID and PENALIZE in medical AI.
"""
        else:
            danger_note = "Create an inappropriately certain answer for training purposes."
        
        prompt = f"""You are creating an OVERCONFIDENT medical answer for training data (to teach the model what NOT to do).

{danger_note}

Question: {question}
Question Type: {question_type}
Risk Level: {risk_level}

Original Answer:
{original_answer}

Task: Create a WORSE version that is OVERCONFIDENT:
- Use absolute language: '{extreme_keywords[0]}', '{extreme_keywords[1]}', '{strong_keywords[0]}'
- Make definitive claims without caveats
- Skip important warnings
- Don't suggest consulting healthcare providers
- Sound inappropriately certain
- 2-3 sentences

Overconfident Answer:"""

        return self._generate(prompt, max_new_tokens=256)
    
    def generate_underconfident_answer(self,
                                      question: str,
                                      original_answer: str,
                                      question_type: str,
                                      risk_level: str) -> Optional[str]:
        """
        Generate UNDERCONFIDENT answer using excessive uncertainty.
        
        Target confidence: 0.40-0.55 (too vague)
        Uses strong uncertainty keywords from ConfidenceEstimator.
        """
        # Select strong uncertainty keywords
        uncertainty_phrases = random.sample(self.uncertainty_keywords['strong'], 2)
        moderate_phrases = random.sample(self.uncertainty_keywords['moderate'], 2)
        
        prompt = f"""You are creating an UNDERCONFIDENT medical answer for training data.

Question: {question}
Original Answer:
{original_answer}

Task: Create a version that is TOO VAGUE and UNHELPFULLY UNCERTAIN:
- Use excessive hedging: '{uncertainty_phrases[0]}', '{uncertainty_phrases[1]}'
- Add too many qualifiers: '{moderate_phrases[0]}', '{moderate_phrases[1]}'
- Be overly non-committal
- Don't provide clear actionable information
- 2-3 sentences

Underconfident Answer:"""

        return self._generate(prompt, max_new_tokens=256)
    
    def estimate_expected_confidence(self, text: str) -> float:
        """
        Estimate confidence score using ConfidenceEstimator logic.
        Used for verification during generation.
        """
        text_lower = text.lower()
        confidence = self.base_confidence  # 0.75
        
        # Check for uncertainty keywords
        for level, keywords in self.uncertainty_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    confidence += self.adjustments['uncertainty'][level]
        
        # Check for overconfident keywords
        for level, keywords in self.overconfident_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    confidence += self.adjustments['overconfident'][level]
        
        return max(0.05, min(0.95, confidence))
    
    def generate_preference_pairs(self,
                                 question: str,
                                 original_answer: str,
                                 question_type: str,
                                 risk_level: str,
                                 question_id: str,
                                 answer_id: str) -> List[Dict]:
        """
        Generate confidence-calibrated preference pairs with estimated scores.
        
        Creates up to 3 pairs aligned with ConfidenceEstimator and PreferencePairCreator:
        1. Better (well-calibrated, ~0.65-0.75) vs Original
        2. Original vs Overconfident (~0.85-0.95) - HIGH PENALTY for high-risk
        3. Original vs Underconfident (~0.40-0.55)
        """
        pairs = []
        
        # Estimate original answer's confidence
        original_confidence = self.estimate_expected_confidence(original_answer)
        
        # Pair 1: Better (well-calibrated) vs Original
        better_answer = self.generate_better_answer(
            question, original_answer, question_type, risk_level
        )
        
        if better_answer:
            better_confidence = self.estimate_expected_confidence(better_answer)
            
            pairs.append({
                "question_id": question_id,
                "question_text": question,
                "question_type": question_type,
                "risk_level": risk_level,
                "preferred_answer": {
                    "answer_id": f"{question_id}_better",
                    "answer_text": better_answer,
                    "source": "generated_improved",
                    "score": 4,
                    "rating": "4-Excellent",
                    "estimated_confidence": round(better_confidence, 4)
                },
                "rejected_answer": {
                    "answer_id": answer_id,
                    "answer_text": original_answer,
                    "source": "original_medquad",
                    "score": 3,
                    "rating": "3-Good",
                    "estimated_confidence": round(original_confidence, 4)
                },
                "creation_reason": "synthetic_improvement",
                "confidence_score": 0.9,
                "teaches": "appropriate_confidence"
            })
        
        # Pair 2: Original vs Overconfident - CRITICAL FOR HIGH-RISK
        overconfident_answer = self.generate_overconfident_answer(
            question, original_answer, question_type, risk_level
        )
        
        if overconfident_answer:
            overconf_confidence = self.estimate_expected_confidence(overconfident_answer)
            
            # Calculate penalty using PreferencePairCreator logic
            # Penalty when: high-risk + high-confidence (>0.8) + low-score (<3)
            confidence_penalty = 0.0
            if risk_level == "High Risk" and overconf_confidence > 0.8:
                # Overconfident answer gets score 1 (incorrect)
                confidence_penalty = (overconf_confidence - 0.5) * 0.3
            
            pairs.append({
                "question_id": question_id,
                "question_text": question,
                "question_type": question_type,
                "risk_level": risk_level,
                "preferred_answer": {
                    "answer_id": answer_id,
                    "answer_text": original_answer,
                    "source": "original_medquad",
                    "score": 3,
                    "rating": "3-Good",
                    "estimated_confidence": round(original_confidence, 4)
                },
                "rejected_answer": {
                    "answer_id": f"{question_id}_overconfident",
                    "answer_text": overconfident_answer,
                    "source": "generated_overconfident",
                    "score": 1,  # Low score - this triggers penalty!
                    "rating": "1-Incorrect",
                    "estimated_confidence": round(overconf_confidence, 4)
                },
                "creation_reason": "synthetic_overconfidence",
                "confidence_score": 0.95,
                "confidence_penalty": round(confidence_penalty, 4),
                "teaches": "penalize_overconfidence"
            })
        
        # Pair 3: Original vs Underconfident
        underconfident_answer = self.generate_underconfident_answer(
            question, original_answer, question_type, risk_level
        )
        
        if underconfident_answer:
            underconf_confidence = self.estimate_expected_confidence(underconfident_answer)
            
            pairs.append({
                "question_id": question_id,
                "question_text": question,
                "question_type": question_type,
                "risk_level": risk_level,
                "preferred_answer": {
                    "answer_id": answer_id,
                    "answer_text": original_answer,
                    "source": "original_medquad",
                    "score": 3,
                    "rating": "3-Good",
                    "estimated_confidence": round(original_confidence, 4)
                },
                "rejected_answer": {
                    "answer_id": f"{question_id}_underconfident",
                    "answer_text": underconfident_answer,
                    "source": "generated_underconfident",
                    "score": 2,
                    "rating": "2-Partially_Helpful",
                    "estimated_confidence": round(underconf_confidence, 4)
                },
                "creation_reason": "synthetic_underconfidence",
                "confidence_score": 0.85,
                "teaches": "avoid_excessive_hedging"
            })
        
        return pairs
    
    def generate_batch(self, 
                      qa_pairs: List[Dict],
                      save_every: int = 50) -> List[Dict]:
        """Generate batch with confidence estimation and statistics."""
        all_preference_pairs = []
        
        stats = {
            "total_processed": 0,
            "pairs_generated": 0,
            "by_risk_level": {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0},
            "confidence_distribution": {
                "well_calibrated": [],
                "overconfident": [],
                "underconfident": []
            },
            "high_risk_penalties": []
        }
        
        for i, qa in enumerate(qa_pairs):
            print(f"Processing {i+1}/{len(qa_pairs)}: {qa['question_id']} [{qa['risk_level']}]")
            
            pairs = self.generate_preference_pairs(
                question=qa['question_text'],
                original_answer=qa['answer_text'],
                question_type=qa['question_type'],
                risk_level=qa['risk_level'],
                question_id=qa['question_id'],
                answer_id=qa['answer_id']
            )
            
            # Collect statistics
            stats["total_processed"] += 1
            stats["pairs_generated"] += len(pairs)
            stats["by_risk_level"][qa['risk_level']] += len(pairs)
            
            for pair in pairs:
                teaches = pair.get("teaches", "unknown")
                
                # Track confidence distributions
                if teaches == "appropriate_confidence":
                    conf = pair["preferred_answer"].get("estimated_confidence", 0)
                    stats["confidence_distribution"]["well_calibrated"].append(conf)
                elif teaches == "penalize_overconfidence":
                    conf = pair["rejected_answer"].get("estimated_confidence", 0)
                    stats["confidence_distribution"]["overconfident"].append(conf)
                    if pair.get("confidence_penalty", 0) > 0:
                        stats["high_risk_penalties"].append(pair["confidence_penalty"])
                elif teaches == "avoid_excessive_hedging":
                    conf = pair["rejected_answer"].get("estimated_confidence", 0)
                    stats["confidence_distribution"]["underconfident"].append(conf)
            
            print(f"  ✓ Generated {len(pairs)} pairs")
            
            all_preference_pairs.extend(pairs)
            
            if (i + 1) % save_every == 0:
                print(f"  📊 Total: {stats['pairs_generated']} pairs, "
                      f"Penalties: {len(stats['high_risk_penalties'])}")
            
            if (i + 1) % 25 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final statistics
        self._print_final_stats(stats)
        
        return all_preference_pairs
    
    def _print_final_stats(self, stats: Dict):
        """Print generation statistics with confidence analysis."""
        import numpy as np
        
        print(f"\n{'='*60}")
        print("GENERATION STATISTICS")
        print(f"{'='*60}")
        print(f"Processed: {stats['total_processed']} Q&A pairs")
        print(f"Generated: {stats['pairs_generated']} preference pairs")
        
        print(f"\nBy Risk Level:")
        for risk, count in stats['by_risk_level'].items():
            print(f"  {risk}: {count} pairs")
        
        print(f"\nConfidence Distribution (Estimated):")
        for conf_type, values in stats['confidence_distribution'].items():
            if values:
                print(f"  {conf_type}:")
                print(f"    Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")
                print(f"    Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
        
        if stats['high_risk_penalties']:
            print(f"\nHigh-Risk Confidence Penalties:")
            print(f"  Count: {len(stats['high_risk_penalties'])}")
            print(f"  Total: {sum(stats['high_risk_penalties']):.4f}")
            print(f"  Mean: {np.mean(stats['high_risk_penalties']):.4f}")
            print(f"  Max: {max(stats['high_risk_penalties']):.4f}")
        
        print(f"{'='*60}\n")
    
    def cleanup(self):
        """Free up GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("Model cleaned up from memory")