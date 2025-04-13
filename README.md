# Reproducing a GPT-Like Model on Lyrics Data

This project implements a character-level Transformer-based language model inspired by GPT, trained from scratch using PyTorch. The primary goal was to gain a deeper understanding of the **foundational architecture behind GPT models (Transformers)** by building and training one end-to-end.

Training was conducted on a **Lambda Labs A10 GPU (24 GB)** to ensure fast iteration, efficient experimentation, and full reproducibility of model behavior and training curves.

---

## 📌 Project Overview

- **Model Type:** GPT-style Transformer (char-level)
- **Framework:** PyTorch
- **Architecture:** 6 layers, 6 heads, 384-dim embeddings (~10.80M parameters)
- **Objective:** Learn the structure and style of music lyrics through next-character prediction
- **Platform:** Lambda Labs A10 GPU instance (California region)

---

## 🧠 Dataset Evolution

### ✅ Phase 1: `drake-lyrics.txt`

The model was initially trained on a small `.txt` file of Drake lyrics.

**Problem:**  
After ~2000 training steps, validation loss began to increase while training loss continued to drop — a clear sign of **overfitting** due to the limited dataset size.

### ✅ Phase 2: `spotify_millsongdata.csv`

To mitigate overfitting, we switched to the **Spotify Million Song Dataset**, which contains a large corpus of lyrics across artists and genres.

- Lyrics were extracted from the `text` column.
- Data was shuffled to increase diversity.
- The full corpus was joined into a single training string for character-level modeling.

**Result:**  
With this richer dataset, the model achieved significantly better generalization:
- **Final train acc:** ~69.6%
- **Final val acc:** ~69.5%
- Validation and training losses stayed tightly coupled throughout training — no overfitting observed even at 6000 steps.

---

## 🧱 Model Architecture

- **Embedding size:** 384
- **Heads:** 6
- **Layers:** 6
- **Dropout:** 0.2
- **Context window (`block_size`):** 256
- **Optimizer:** AdamW
- **Training steps:** 6000
- **Eval interval:** Every 500 steps
- **Evaluation metric:** Cross-entropy loss + character-level accuracy

---

## 🔬 Evaluation Results

| Step | Train Loss | Val Loss | Train Acc | Val Acc |
|------|------------|----------|-----------|----------|
| 0    | 4.4181     | 4.4173   | 0.0037    | 0.0037   |
| 1000 | 1.3376     | 1.3351   | 0.5979    | 0.5987   |
| 3000 | 1.0751     | 1.0781   | 0.6773    | 0.6769   |
| 5999 | **1.0033** | **1.0092** | **0.6964** | **0.6955** |

---

## 📊 Loss & Accuracy Curves

Plots were generated using `matplotlib` to visualize training progress and overfitting. Both loss and accuracy improved steadily across all 6000 steps.

---

## 🚀 Summary

This project demonstrates a full-stack, from-scratch implementation of a GPT-style model, covering:
- Architecture design
- Data processing
- Overfitting mitigation
- Loss/accuracy tracking
- Infrastructure and scaling on GPU

It also highlights how dataset choice plays a critical role in generalization quality and model performance.

---
