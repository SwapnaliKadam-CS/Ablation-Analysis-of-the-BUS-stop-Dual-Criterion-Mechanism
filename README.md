# 🚀 Ablation Analysis of the BUS-stop Dual-Criterion Mechanism

## 💡 Project Overview

This repository contains the code for the **ablation study of the BUS-stop** early stopping methodology. The original BUS-stop method employs a dual-criterion mechanism—**Confidence Similarity ($\text{S}_{\text{conf}}$)** for stability and **Class Distribution Similarity ($\text{S}_{\text{class}}$)** for checkpoint selection—on unlabeled data.

Our primary hypothesis is that this dual-criterion approach introduces unnecessary computational overhead, and a simplified **Confidence Similarity Only (CS-Only)** variant can achieve comparable performance (within 1% point) with 50% fewer metric calculations per epoch.


### 🔑 Key Findings Summary

The following table summarizes the performance metrics for the four tested early stopping strategies, directly supporting the hypothesis that the CS-Only approach is both efficient and accurate.

| Model Name | Stop Metric | Save Metric | Test Acc. | Total Epochs |
| :--- | :--- | :--- | :--- | :--- |
| **Combined (BUS)** | $\text{S}_{\text{conf}}$ | $\text{S}_{\text{class}}$ (queue avg) | **0.8611** | 15 |
| **CS-Only (Ablated)** | $\text{S}_{\text{conf}}$ | $\min \text{S}_{\text{conf}}$ | **0.8611** | 15 |
| **CDS-Only (Ablated)** | $-\text{S}_{\text{class}}$ | $\max \text{S}_{\text{class}}$ | 0.7490 | 6 |
| **Standard (Val)** | Val Loss | $\min$ Val Loss | 0.8358 | 11 |

**Analysis:** Experiments on the SST-2 dataset partially support this hypothesis: the
CS-Only variant matches the Combined (BUS) model’s peak
accuracy (0.8611), confirming the performance goal, while the
CDS-Only variant fails significantly (0.7490). This validates CS
as the primary convergence driver but reveals CDS is essential
for robust checkpoint selection. We conclude that CS alone
suffices for stopping decisions, though CDS should be retained
for checkpoint validation. This simplification reduces per-epoch
metric calculations by 50% without compromising generalization
performance, offering substantial efficiency gains for resource-
constrained fine-tuning scenarios.

## 💻 Requirements and Installation

This project was developed and executed entirely within a **Google Colab** environment, which is the recommended method for reproduction.

Collab Link - https://colab.research.google.com/drive/1i1Uoo1B7ec1Faf8QZ43IJN5Rq_O6a6y6?usp=sharing

### 1. Clone the Repository (Cell 1)

Clone the original BUS-stop repository which serves as the base code, and navigate to the Keras implementation directory.

```bash
# 1. Clone the base BUS-stop repository
git clone [https://github.com/DMCB-GIST/BUS-stop.git](https://github.com/DMCB-GIST/BUS-stop.git)
%cd BUS-stop/bus-stop-keras

# 2.  Install Dependencies (Cell 1)
The following dependencies are required:
!pip install -q tensorflow tf_keras
!pip install -q transformers==4.44.0 scikit-learn pandas sentencepiece datasets

# 3. Data and Model Setup (Cell 1)
This script downloads the $\text{BERT}_{\text{base-uncased}}$ model weights and generates the balanced SST-2 data splits (200 labeled samples) used for the main ablation study.
Bash# Ensure BERT files are downloaded and data splits are created
!rm -rf params/bert_base # Force fresh download
!python setup_experiments.py

# 4. Usage: Running the Ablation Study
The core experiments are run using the modified train_engine function (defined in the accompanying notebook/scripts) to execute the four model variants (Combined, Conf Only, Class Only, and Standard Baseline).

#Execution Command (Cell 3)
The commands below execute the primary ablation study:

# Assuming train_engine and Args are defined:
DEFAULT_SEED = 42

print("EXPERIMENT 1/4: Original BUS-stop (Combined)")
acc_orig, _, _, _ = train_engine("Combined", mode='combined', seed=DEFAULT_SEED)

print("\nEXPERIMENT 2/4: Confidence Similarity Only")
acc_conf, _, _, _ = train_engine("Conf Only", mode='conf', seed=DEFAULT_SEED)

print("\nEXPERIMENT 3/4: Class Distribution Only")
acc_class, _, _, _ = train_engine("Class Only", mode='class', seed=DEFAULT_SEED)

print("\nEXPERIMENT 4/4: Standard Validation (Baseline)")
acc_std, _, _, _ = train_engine("Standard", mode='standard', val_ratio=0.1, seed=DEFAULT_SEED)

# 5. Visualization and Validation (Cells 4, 5, 6)
The remaining cells handle analysis and visualization:
Cell 4: Generates the Confusion Matrices for all four models.

Cell 5: Generates the epoch-by-epoch Training Dynamics plots (Accuracy, Stop Metric, Save Metric).

Cell 6: Executes the Robustness Check by running the core experiments with an alternate random seed to ensure stability.

Cell 7: Executes the Large-Scale Data Efficiency analysis to generate Figure 6.

#🔗 Original Source Repository
This project is an ablation analysis built upon the original work:Original Code: $\text{\url{https://github.com/DMCB-GIST/BUS-stop}}$Ablation Analysis Code: $\text{\url{https://github.com/SwapnaliKadam-CS/Ablation-Analysis-of-the-BUS-stop-Dual-Criterion-Mechanism}}$