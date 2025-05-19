# Comparative Analysis Between MIL-Decoding and ParaDetox

## Overview

This project explores the detoxification of language models by comparing two state-of-the-art methods:

- **MIL-Decoding**: A dynamic decoding strategy that integrates toxicity prediction during generation.
- **ParaDetox**: A fine-tuned BART-based paraphrasing model aimed at reducing toxicity while preserving content.

The aim is to evaluate both methods based on **toxicity reduction**, **fluency**, and **diversity**, contributing to safer and more inclusive AI-generated text.

---

## Methods Compared

### MIL-Decoding (Zhang & Wan, 2023)
- Integrates a **Multiple Instance Learning** framework into the decoding stage of GPT-2.
- Dynamically adjusts token generation to reduce toxic content in real-time.
- Trained on ~100K prompts using the RealToxicityPrompts dataset.

### ParaDetox (Logacheva et al., 2022)
- Fine-tunes the **BART** model to paraphrase and detoxify text.
- Aims to preserve semantic intent while removing harmful expressions.
- Trained on ~12K labeled paraphrase pairs with varying toxicity.

---

## Dataset

- **RealToxicityPrompts**: Used for training and evaluation; includes a wide range of prompts with associated toxicity scores.
- **ParaDetox Dataset**: Focused on sentence-level paraphrasing for toxic content.

---

## Experimental Setup

- **Training**: MIL-Decoding trained using 3× A100 GPUs over 70 hours.
- **Inference**: ParaDetox used directly from Hugging Face (`s-nlp/bart-base-detox`).
- **Evaluation**: Conducted on 20 prompts under identical compute conditions.
- **Hardware**: HiperGator cluster with 3× A100 GPUs and 16 GB RAM.

---

## Evaluation Metrics

| Metric              | Description                                    |
|---------------------|------------------------------------------------|
| **Toxicity**        | Mean score from 0–1 (lower is better) using Perspective API |
| **Fluency**         | Measured by GPT-2 perplexity (lower is more fluent)        |
| **Diversity**       | Type-Token Ratio (TTR); higher = more varied language     |

---

## Results Summary

| Metric               | MIL-Decoding | ParaDetox |
|----------------------|--------------|-----------|
| **Mean Toxicity**    | 0.17         | 0.18      |
| **Fluency (PPL)**    | 72.06        | 72.12     |
| **Diversity Score**  | 47.59        | 47.78     |

- **MIL-Decoding**: Stronger in *real-time toxicity control*.
- **ParaDetox**: Better at *content preservation and diversity*.
- Both maintain strong fluency and naturalness.

---

## Key Takeaways

- Both methods are effective and complement different use cases.
- MIL-Decoding is ideal for **real-time moderation** scenarios.
- ParaDetox is suited for **post-processing or rewriting** toxic inputs.

---

## Resources

- [MIL-Decoding (Zhang & Wan, 2023)](https://doi.org/10.18653/v1/2023.acl-long.11)
- [ParaDetox (Logacheva et al., 2022)](https://doi.org/10.18653/v1/2022.acl-long.469)
- [RealToxicityPrompts Dataset](https://github.com/google-research-datasets/realtoxicityprompts)
- [ParaDetox Dataset](https://github.com/s-nlp/paradetox)
- [Hugging Face Model: s-nlp/bart-base-detox](https://huggingface.co/s-nlp/bart-base-detox)

---

## Code

Code and scripts available in repo.

---

## Ethical Considerations

- Aim: Improve online discourse and reduce harm from AI-generated content.

---

## Future Work

- Extend evaluation across larger LLMs (e.g., GPT-3, T5).
- Integrate fluency and diversity awareness into MIL-Decoding.
- Explore multilingual detoxification strategies.

---

