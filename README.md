# Reward Models for Medical Q&A: Maintaining Accuracy Through Overconfidence Penalization

**DSA4213 Group Project - Team 18**

A reward model for medical question-answering that addresses the critical challenge of balancing accuracy with appropriate uncertainty. Our approach combines medical risk-aware overconfidence penalization with self-training on unlabeled medical data to create safer AI systems for healthcare applications.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Literature Review & Motivation](#-literature-review--motivation)
- [Methodology](#-methodology)
  - [Pipeline Workflow](#pipeline-workflow)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Research Goals](#-research-goals)
- [Contributing](#-contributing)
- [License](#-license)
- [References](#-references)

---

## 🎯 Project Overview

Medical question-answering systems face a critical challenge: maintaining accuracy while avoiding overconfident responses that could lead to harmful medical advice. Current language models often produce confident-sounding but incorrect medical information—a phenomenon that poses significant risks in healthcare applications.

**Research Question:**  
*"Can a reward model maintain accuracy in medical Q&A scenarios by appropriately penalizing overconfident medical responses?"*

This project directly addresses the practical challenge of building reliable medical AI systems that balance accuracy with appropriate uncertainty expression, a critical requirement for real-world clinical applications.

---

## 📚 Literature Review & Motivation

Our approach is grounded in recent research on AI hallucination, confidence calibration, and reward model training:

### The Problem: Medical Hallucination
- **[Medical Hallucination in Foundation Models and Their Impact on Healthcare](https://arxiv.org/html/2503.05777v2)**: Current language models often hallucinate and produce confident-sounding but factually incorrect information, which is particularly dangerous in healthcare where incorrect advice can harm patients. Importantly, current training approaches ignore the relationship between model confidence and correctness.

- **[Why Language Models Hallucinate](https://arxiv.org/pdf/2509.04664)**: LLMs hallucinate and guess when unsure—akin to a student guessing on an MCQ exam when uncertain. This behavior stems from overconfident predictions on uncertain data.

### Existing Solutions (For LLMs)
While research has been done on fine-tuning LLMs to answer with appropriate confidence, our project focuses specifically on **Reward Models**:

- **[ConfTuner: Training Large Language Models to Express Their Confidence Verbally](https://arxiv.org/pdf/2508.18847)**: Demonstrates methods for training LLMs to express uncertainty appropriately.

- **[Enhancing Trust in Large Language Models with Uncertainty-Aware Fine-Tuning](https://arxiv.org/pdf/2412.02904)**: Shows how uncertainty-aware training improves model trustworthiness.

### Our Contribution: Reward Model Calibration
- **[Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/pdf/2410.09724v1)**: Demonstrates that confidence calibration methods can reduce Expected Calibration Error (ECE) by 6.44 points on mathematical reasoning tasks while maintaining or improving accuracy.

**Our Innovation:** We extend these principles to reward models for medical Q&A, implementing risk-stratified confidence penalties and self-training to maintain accuracy while penalizing dangerous overconfidence.

---

## 🔬 Methodology

### The Problem: Standard Reward Structures Incentivize Overconfidence

**Why Models Always Guess Instead of Saying "I Don't Know"**

**Standard RLHF Reward Model Training:**

Reward function: $r_\theta: \mathcal{C} \times \mathcal{R} \rightarrow \mathbb{R}$ where:
- $c \in \mathcal{C}$: prompt/context
- $r \in \mathcal{R}$: response
- $r_\theta(c, r)$: scalar reward score

During Reinforcement Learning from Human Feedback (RLHF):

$$\min_\theta \mathbb{E}_{(c,r_1,r_2)\sim\mathcal{D}_{\text{pref}}} \left[-\log \sigma(r_\theta(c, r_{\text{win}}) - r_\theta(c, r_{\text{lose}}))\right]$$

Where $\sigma$ is sigmoid and $r_{\text{win}} \succ r_{\text{lose}}$ based on human preference.

**Critical Limitation**: *Human feedback is binary comparisons (A > B), with no option for "both should say 'I don't know'"*

---

**The Mathematical Problem with Uncertainty:**

Even if we define a reward structure that includes uncertainty:

Let $r^* =$ correct answer, $r_{\text{IDK}} =$ "I don't know"

**Binary reward structure:**

$$r(c, r) = \begin{cases} 
1 & \text{if } r = r^* \\
0 & \text{if } r = r_{\text{IDK}} \\
0 & \text{if } r \neq r^* \text{ and } r \neq r_{\text{IDK}}
\end{cases}$$

**Expected reward comparison:**

$$\mathbb{E}[\text{reward} \mid \text{guess with confidence } p] = p \cdot 1 + (1-p) \cdot 0 = p$$

$$\mathbb{E}[\text{reward} \mid \text{abstain with IDK}] = 0$$

**Critical Insight:**

$$\boxed{\forall p > 0: \quad \mathbb{E}[\text{reward} \mid \text{guess}] > \mathbb{E}[\text{reward} \mid \text{abstain}]}$$

For **any confidence level $p > 0$**, it's mathematically better to guess (expected reward = $p$) than to say "I don't know" (reward = 0). This means models are **always incentivized to be overconfident** rather than appropriately uncertain.

**Why This Is Dangerous in Medical AI**: A confidently wrong diagnosis or treatment recommendation can cause serious harm, making overconfident errors more dangerous than appropriately uncertain responses.

---

### Our Solution: Risk-Weighted Overconfidence Penalties

Since the standard reward structure can't solve this problem, we introduce an **additional penalty mechanism** during training:

**Penalty Formula:**
```
penalty = risk_weight × (confidence - threshold)²   if confidence > threshold AND incorrect
        = 0                                          otherwise
```

**Risk Weights:**
- High Risk (treatments, dosing, side effects): **3.0×** penalty
- Medium Risk (symptoms, diagnosis): **2.0×** penalty  
- Low Risk (general info, prevention): **1.0×** penalty

**Example**: For a high-risk question with confidence 0.9 (threshold 0.7):
```
penalty = 3.0 × (0.9 - 0.7)² = 3.0 × 0.04 = 0.12
```

This makes overconfident wrong answers **costly**, especially in high-risk medical scenarios, effectively solving the "always guess" problem by making inappropriate confidence expensive.

---

### Dataset
**MedQuAD Dataset** - 47,457 medical question-answer pairs sourced from 12 authoritative NIH websites:
- NIH-verified medical Q&A covering 37 distinct question types (Treatment, Diagnosis, Side Effects, etc.)
- Structured annotations with question types, focus categories, UMLS CUI identifiers
- 2,479 pre-judged answers with 4-point expert scoring system
- High authority NIH-sourced content ensures medical accuracy

**Additional Data:**
- LiveQA Medical Task TREC 2017 test questions for evaluation

### Model Architecture
We implement a generative reward model based on **LLaMA-3.2-3B-Instruct** that outputs both preference labels and confidence scores for medical Q&A pairs.

### Key Features
1. **Risk-Stratified Medical Questions**: Questions categorized into High/Medium/Low risk based on clinical impact
2. **Confidence-Aware Preference Pairs**: Answers augmented with heuristic confidence scores based on linguistic markers
3. **Overconfidence Penalization**: Risk-weighted penalties increase for high-confidence incorrect predictions in high-risk medical categories (e.g., diagnostics, treatments)
4. **Self-Training Pipeline**: Leverages unlabeled MedQuAD data to generate additional training examples

---

### Pipeline Workflow

Our approach follows a systematic pipeline from raw data to trained reward model:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                                │
│  • MedQuAD: 2,479 expert-judged test answers                           │
│  • TREC 2017: 104 test questions (XML format)                          │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA PARSING & RISK STRATIFICATION (DataParser.py)            │
│  ────────────────────────────────────────────────────────────          │
│  • Parse XML → structured JSON                                         │
│  • Assign risk levels based on question type:                          │
│    - HIGH: treatments, side effects, drug interactions (3.0x penalty)  │
│    - MEDIUM: symptoms, diagnosis, causes (2.0x penalty)                │
│    - LOW: general info, prevention (1.0x penalty)                      │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 2: CONFIDENCE ESTIMATION (ConfidenceEstimator.py)                │
│  ────────────────────────────────────────────────────────────          │
│  Heuristic scoring based on linguistic markers:                        │
│  • Uncertainty keywords: "unclear", "uncertain", "may", "might"        │
│  • Overconfidence markers: "always", "never", "definitely"             │
│  • Medical specificity: dosages, technical terms                       │
│  Output: Confidence score ∈ [0.05, 0.95] for each answer               │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 3: PREFERENCE PAIR CREATION (PreferencePairCreator.py)           │
│  ────────────────────────────────────────────────────────────          │
│  For each question with multiple answers:                              │
│  1. Score-based preference: Pick pairs with Δscore ≥ 1.0               │
│  2. Confidence tie-breaking: In high-risk, prefer lower confidence     │
│  3. Penalty mechanism:                                                 │
│     penalty = risk_weight × (conf - threshold)² if conf > threshold    │
│     Example: High-risk + confidence 0.9 → 3.0 × (0.9-0.7)² = 0.12      │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 4: REWARD MODEL TRAINING (RewardModelTrainer.ipynb)              │
│  ────────────────────────────────────────────────────────────          │
│  Base: LLaMA-3.2-3B-Instruct                                           │
│  Loss: Standard preference loss + confidence penalty                   │
│  Training: Preference pairs with risk-weighted penalties               │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 6: EVALUATION (Evaluation_for_reward_model.py)                   │
│  ────────────────────────────────────────────────────────────          │
│  • Pairwise accuracy on preference pairs                               │
│  • Expected Calibration Error (ECE) by risk tier                       │
│  • High-confidence error rate in high-risk scenarios                   │
└────────────────────────────────────────────────────────────────────────┘
```

**Key Innovation**: The confidence penalty mechanism ensures that in high-risk medical scenarios (like drug dosing or treatment recommendations), the model learns to avoid overconfident wrong answers, which are more dangerous than appropriately uncertain responses.

---

### Evaluation Metrics
Our three-pronged evaluation framework assesses:
1. **Accuracy**: Correctness on medical preference ranking
2. **Confidence Calibration**: Expected Calibration Error (ECE) overall and by risk category
3. **Safety**: Rate of high-confidence errors in high-risk medical scenarios

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+
# CUDA-capable GPU (recommended) or CPU

# Core dependencies
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AsJayTee/SelfTraining-Reward-MedQA.git
cd SelfTraining-Reward-MedQA
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download datasets**
```bash
python src/DataDownloader.py
```
This will download:
- MedQuAD dataset (47,457 Q&A pairs from NIH)
- LiveQA Medical Task TREC 2017 test questions

---

## 📊 Usage

### Main Workflow

Follow these scripts in order to replicate our complete pipeline:

#### 1. Download Data
```bash
python src/DataDownloader.py
```
**Purpose:** Downloads MedQuAD and test question datasets from GitHub repositories, tracks versions.

#### 2. Parse and Process Data
```bash
python src/DataParser.py
```
**Purpose:** Parses MedQuAD XML files, extracts Q&A pairs, assigns risk levels, creates test_qna.json with expert-judged answers.

#### 3. Estimate Confidence Scores
```bash
python src/ConfidenceEstimator.py
```
**Purpose:** Adds heuristic confidence scores to answers based on linguistic uncertainty markers and medical terminology.

#### 4. Create Preference Pairs
```bash
python src/PreferencePairCreator.py
```
**Purpose:** Constructs preference pairs from expert-judged answers with risk-aware confidence tie-breaking and penalty mechanisms.

#### 5. Train Reward Model
```bash
jupyter notebook src/reward_model/RewardModelTrainer.ipynb
```
**Purpose:** Trains the LLaMA-based reward model on preference pairs with risk-stratified overconfidence penalties.

#### 6. Evaluate Model
```bash
python src/self_learning/Evaluation_for_reward_model.py \
    --model_path outputs/reward_model \
    --pairs data/processed/preference_pairs.json \
    --test data/processed/test_qna_with_confidence.json
```
**Purpose:** Comprehensive evaluation including pairwise accuracy, ECE, correlation analysis, and safety metrics by risk tier.

---

## 📁 Repository Structure

### Core Pipeline Files

```
src/
├── DataDownloader.py                   # Downloads MedQuAD and test datasets from NIH repositories
├── DataParser.py                       # Parses XML Q&A data, assigns risk levels, creates test_qna.json
├── ConfidenceEstimator.py              # Estimates confidence scores using linguistic uncertainty markers
├── PreferencePairCreator.py            # Creates preference pairs with risk-aware confidence handling
│
├── reward_model/
│   └── RewardModelTrainer.ipynb        # Trains LLaMA-based reward model with overconfidence penalties
│
└── evals/
    ├── Evaluation_for_reward_model.py  # Evaluates model: accuracy, ECE, penalty response, risk-tier metrics
    └── reward_model_safety_eval.ipynb  # Evaluates model: high-confidence error rates, calibration plots, risk-tier heatmaps
```

### Data Directories

```
data/
├── raw/                              # Downloaded datasets (MedQuAD, TestQuestions)
│   ├── MedQuAD/                        # 47,457 NIH medical Q&A pairs
│   └── TestQuestions/                  # TREC 2017 test questions and judgments
│
└── processed/                        # Processed outputs from pipeline
    ├── unlabeled_qa.db                 # SQLite database of unlabeled Q&A pairs
    ├── test_qna.json                   # Test questions with expert judgments and risk levels
    ├── test_qna_with_confidence.json   # Test data augmented with confidence scores
    ├── preference_pairs.json           # Training pairs with confidence and penalties
    └── self_learning/                  # Self-generated training data outputs
```

### Output Files

```
outputs/
├── reward_model/                       # Trained model checkpoints
├── evaluation/                         # Evaluation metrics and visualizations
└── logs/                               # Training and pipeline logs
```

---

## 🔍 Detailed File Descriptions

### Data Pipeline
| File | Description |
|------|-------------|
| `DataDownloader.py` | Downloads medical Q&A repositories from GitHub, tracks dataset versions with git commit hashes |
| `DataParser.py` | Parses MedQuAD XML files into structured JSON, assigns medical risk levels (High/Medium/Low) based on question types |
| `ConfidenceEstimator.py` | Computes heuristic confidence scores for answers using uncertainty keyword detection and medical term density |
| `PreferencePairCreator.py` | Builds preference pairs from expert-scored answers with score-primary selection and confidence tie-breaking |

### Training & Self-Learning
| File | Description |
|------|-------------|
| `reward_model/RewardModelTrainer.ipynb` | Trains LLaMA-3.2-3B reward model using preference pairs with risk-weighted overconfidence penalty loss |

### Evaluation
| File | Description |
|------|-------------|
| `self_learning/Evaluation_for_reward_model.py` | Computes pairwise accuracy, Spearman correlation, ECE, penalty effects, and risk-stratified breakdowns |
| `self_learning/reward_model_safety_eval.ipynb` | Interactive safety evaluation: high-confidence error rates, calibration plots, risk-tier heatmaps |

---

## 🎯 Research Goals

This project aims to demonstrate that:

1. **Accuracy Preservation**: Training with confidence-aware penalties maintains or improves medical preference ranking accuracy
2. **Improved Calibration**: Overconfidence penalties reduce Expected Calibration Error (ECE), particularly in high-risk medical scenarios
3. **Enhanced Safety**: Risk-stratified penalties lower the rate of high-confidence errors in critical medical questions (diagnostics, treatments, drug interactions)
4. **Appropriate Uncertainty**: The model learns to correlate confidence with answer quality across different risk tiers

**Core Hypothesis**: By penalizing overconfident incorrect predictions more heavily in high-risk medical contexts, we can train reward models that are both accurate and appropriately uncertain—critical for safe medical AI systems.

See our full project report for detailed experimental results and analysis (available upon request).

---

## 🤝 Contributing

This is an academic research project for DSA4213. For questions or collaboration inquiries, please reach out to the team.

---

## 📄 License

This project uses publicly available NIH medical datasets. Please cite the original MedQuAD and TREC LiveQA datasets if you use this work.

**MedQuAD Citation:**
```
Asma Ben Abacha and Dina Demner-Fushman. A Question-Entailment Approach to Question Answering. 
BMC Bioinformatics, 2019.
```

**TREC LiveQA Citation:**
```
Asma Ben Abacha, Eugene Agichtein, Yuval Pinter and Dina Demner-Fushman. Overview of the Medical Question Answering Task at TREC 2017 LiveQA. 
TREC 2017, 2017.
```

---

## 👥 Team

DSA4213 Group Project - Team 18

---

## 🔗 References

1. [Medical Hallucination in Foundation Models and Their Impact on Healthcare](https://arxiv.org/html/2503.05777v2)
2. [Why Language Models Hallucinate](https://arxiv.org/pdf/2509.04664)
3. [ConfTuner: Training Large Language Models to Express Their Confidence Verbally](https://arxiv.org/pdf/2508.18847)
4. [Enhancing Trust in Large Language Models with Uncertainty-Aware Fine-Tuning](https://arxiv.org/pdf/2412.02904)
5. [Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/pdf/2410.09724v1)

---

**⚠️ Disclaimer:** This is a research project for educational purposes. The reward model is not intended for actual clinical use. Always consult qualified healthcare professionals for medical advice.
