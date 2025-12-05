🚀 Ablation Analysis of the BUS-stop Dual-Criterion Mechanism
💡 Project Overview

This repository contains the code and analysis for the ablation study of the BUS-stop early stopping methodology.
The original BUS-stop method uses a dual-criterion mechanism on unlabeled data:

Confidence Similarity (S<sub>conf</sub>) – measures prediction stability

Class Distribution Similarity (S<sub>class</sub>) – used for checkpoint selection

🎯 Hypothesis

The dual-criterion approach may introduce unnecessary computational overhead, and a simplified Confidence-Similarity-Only (CS-Only) variant might:

Achieve accuracy within 1% of the original BUS-stop

Cut per-epoch metric calculations by 50%

🔑 Key Findings Summary
Model Name	Stop Metric	Save Metric	Test Acc.	Total Epochs
Combined (BUS)	S<sub>conf</sub>	S<sub>class</sub> (queue avg)	0.8611	15
CS-Only (Ablated)	S<sub>conf</sub>
min Sconf	S<sub>conf</sub>	0.8611	15
CDS-Only (Ablated)	–	S<sub>class</sub>
max Sclass	0.7490	6
Standard (Val)	Validation Loss	min Val Loss	0.8358	11
📊 Analysis Summary

Experiments on SST-2 partially support the hypothesis:

CS-Only matches the Combined (BUS) model’s best accuracy (0.8611)

CDS-Only performs poorly, confirming it is not reliable alone

CS is the main driver for reliable stopping decisions

CDS is still useful for checkpoint validation

👉 Conclusion:
Confidence Similarity alone is sufficient for stopping, while Class Similarity is helpful but not essential for checkpoint selection—leading to 50% fewer per-epoch metric calculations without sacrificing generalization.

💻 Requirements and Installation

This project is designed and tested entirely using Google Colab, which is the recommended setup.

👉 Colab Notebook:
https://colab.research.google.com/drive/1i1Uoo1B7ec1Faf8QZ43IJN5Rq_O6a6y6?usp=sharing

📥 1. Clone the Repository (Cell 1)
# Clone the base BUS-stop repository
git clone https://github.com/DMCB-GIST/BUS-stop.git
%cd BUS-stop/bus-stop-keras

📦 2. Install Dependencies (Cell 1)
pip install -q tensorflow tf_keras
pip install -q transformers==4.44.0 scikit-learn pandas sentencepiece datasets

📂 3. Data and Model Setup (Cell 1)

Downloads BERT-base-uncased and generates the balanced SST-2 splits (200 labeled samples).

# Force fresh BERT download and generate splits
rm -rf params/bert_base
python setup_experiments.py

🚀 4. Running the Ablation Study

Experiments are executed using the custom train_engine function.

Execution Command (Cell 3)
# Assuming train_engine and Args are defined:
DEFAULT_SEED = 42

print("EXPERIMENT 1/4: Original BUS-stop (Combined)")
acc_orig, _, _, _ = train_engine("Combined", mode='combined', seed=DEFAULT_SEED)

print("\nEXPERIMENT 2/4: Confidence Similarity Only")
acc_conf, _, _, _ = train_engine("Conf Only", mode='conf', seed=DEFAULT_SEED)

print("\nEXPERIMENT 3/4: Class Distribution Only")
acc_class, _, _, _ = train_engine("Class Only", mode='class', seed=DEFAULT_SEED)

print("\nEXPERIMENT 4/4: Standard Validation (Baseline)")
acc_std, _, _, _ = train_engine(
    "Standard",
    mode='standard',
    val_ratio=0.1,
    seed=DEFAULT_SEED
)

📊 5. Visualization & Validation (Cells 4–6)

Cell 4: Confusion matrices for all four models

Cell 5: Training dynamics (Accuracy, Stop Metric, Save Metric)

Cell 6: Robustness check with alternate seed

Cell 7: Large-scale data efficiency analysis (Figure 6)

🔗 Source Repositories

Original BUS-stop Code:
https://github.com/DMCB-GIST/BUS-stop

Ablation Analysis (This Project):
https://github.com/SwapnaliKadam-CS/Ablation-Analysis-of-the-BUS-stop-Dual-Criterion-Mechanism