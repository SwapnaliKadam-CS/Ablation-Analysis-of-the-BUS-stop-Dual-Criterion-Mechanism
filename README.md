# 🚀 Ablation Analysis of the BUS-stop Dual-Criterion Mechanism

## 💡 Project Overview

This repository contains the code for the **ablation study of the BUS-stop** early stopping methodology. [cite_start]The original BUS-stop method employs a dual-criterion mechanism—**Confidence Similarity ($\text{S}_{\text{conf}}$)** for stability and **Class Distribution Similarity ($\text{S}_{\text{class}}$)** for checkpoint selection—on unlabeled data[cite: 6, 17].

[cite_start]Our primary hypothesis is that this dual-criterion approach introduces unnecessary computational overhead, and a simplified **Confidence Similarity Only (CS-Only)** variant can achieve comparable performance (within 1% point) with 50% fewer metric calculations per epoch[cite: 8, 44].

### Key Findings Summary

| Metric | Combined (BUS) | CS-Only | Standard (Val) |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **0.8611** | **0.8611** | 0.8358 |
| **Metric Calcs** | 2 ($\text{S}_{\text{conf}}$, $\text{S}_{\text{class}}$) | 1 ($\text{S}_{\text{conf}}$ only) | 1 (Val Loss) |

[cite_start]The results fully validate the hypothesis: the $\text{CS-Only}$ model successfully matches the original $\text{BUS-stop}$ model's peak accuracy while achieving a **50% reduction in algorithmic complexity**[cite: 9, 47, 87].

## 💻 Requirements and Installation

This project was developed and executed entirely within a **Google Colab** environment, which is the recommended method for reproduction.

Collab Link - https://colab.research.google.com/drive/1i1Uoo1B7ec1Faf8QZ43IJN5Rq_O6a6y6?usp=sharing

### 1. Clone the Repository (Cell 1)

Clone the original BUS-stop repository which serves as the base code, and navigate to the Keras implementation directory.

```bash
# Clone the base BUS-stop repository
git clone [https://github.com/DMCB-GIST/BUS-stop.git](https://github.com/DMCB-GIST/BUS-stop.git)
%cd BUS-stop/bus-stop-keras
