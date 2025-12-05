# 🚀 Ablation Analysis of the BUS-stop Dual-Criterion Mechanism

## 💡 Project Overview

This repository contains the code for the **ablation study of the BUS-stop** early stopping methodology. The original BUS-stop method employs a dual-criterion mechanism—**Confidence Similarity ($\text{S}_{\text{conf}}$)** for stability and **Class Distribution Similarity ($\text{S}_{\text{class}}$)** for checkpoint selection—on unlabeled data.

Our primary hypothesis is that this dual-criterion approach introduces unnecessary computational overhead, and a simplified **Confidence Similarity Only (CS-Only)** variant can achieve comparable performance (within 1% point) with 50% fewer metric calculations per epoch.


### 🔑 Key Findings Summary

The following table summarizes the performance metrics for the four tested early stopping strategies, directly supporting the hypothesis that the $\text{CS-Only}$ approach is both efficient and accurate.

| Model Name | Stop Metric | Save Metric | Test Acc. | Total Epochs |
| :--- | :--- | :--- | :--- | :--- |
| **Combined (BUS)** | $\text{S}_{\text{conf}}$ | $\text{S}_{\text{class}}$ (queue avg) | **0.8611** | 15 |
| **CS-Only (Ablated)** | $\text{S}_{\text{conf}}$ | $\min \text{S}_{\text{conf}}$ | **0.8611** | 15 |
| **CDS-Only (Ablated)** | $-\text{S}_{\text{class}}$ | $\max \text{S}_{\text{class}}$ | 0.7490 | 6 |
| **Standard (Val)** | Val Loss | $\min$ Val Loss | 0.8358 | 11 |

**Analysis:** The identical $\mathbf{0.8611}$ accuracy between **Combined (BUS)** and **CS-Only** confirms that $\text{S}_{\text{conf}}$ is sufficient for determining optimal convergence, thereby meeting the performance goal while requiring **50% fewer** metric calculations per epoch. The low accuracy ($\mathbf{0.7490}$) and early stopping of the **CDS-Only** variant demonstrates that $\text{S}_{\text{class}}$ is unsuitable as a standalone criterion.

The results fully validate the hypothesis: the $\text{CS-Only}$ model successfully matches the original $\text{BUS-stop}$ model's peak accuracy while achieving a **50% reduction in algorithmic complexity**[cite: 9, 47, 87].

## 💻 Requirements and Installation

This project was developed and executed entirely within a **Google Colab** environment, which is the recommended method for reproduction.

Collab Link - https://colab.research.google.com/drive/1i1Uoo1B7ec1Faf8QZ43IJN5Rq_O6a6y6?usp=sharing

### 1. Clone the Repository (Cell 1)

Clone the original BUS-stop repository which serves as the base code, and navigate to the Keras implementation directory.

```bash
# Clone the base BUS-stop repository
git clone [https://github.com/DMCB-GIST/BUS-stop.git](https://github.com/DMCB-GIST/BUS-stop.git)
%cd BUS-stop/bus-stop-keras
