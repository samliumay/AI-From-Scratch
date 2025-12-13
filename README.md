# AI Beyond the Hype: Architecture, Reality, and Risks

> **From Zero to Hero:** A comprehensive roadmap combining theoretical foundations, historical context, implementation, and security risks of Artificial Intelligence.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Active-success)]()

## Overview

This repository is designed for developers, security professionals, and enthusiasts who want to understand AI **beyond the buzzwords**. Unlike standard tutorials, this project bridges the gap between:
1.  **Mathematical Foundations:** (Linear Regression, PCA, Neural Networks)
2.  **Modern Architecture:** (Transformers, LLMs, RAG)
3.  **Real-World Risks:** (OWASP Top 10 for LLMs, Adversarial Attacks)

It includes detailed lecture slides and a growing collection of python implementations starting from scratch.

---

## Content Structure

The repository flows through the evolution of AI, fitting John McCarthy's definition: *"The science and engineering of making machines that can perform tasks that would normally require human intelligence."*

### Part 1: Foundations & History
* **The Origins:** From Gauss (1809) to Turing (1950).
* **Classical AI vs. ML vs. DL:** Understanding the hierarchy.
* **Core Algorithms:** Linear/Logistic Regression, K-NN, PCA.

### Part 2: The Deep Learning Boom
* **Neural Networks:** Perceptrons, Backpropagation, and Hidden Layers.
* **Architectures:** CNNs (AlexNet), GANs, and the shift to Transformers.

### Part 3: Modern Era & LLMs
* **Transformers:** "Attention Is All You Need", Tokenization, Embeddings.
* **Paradigms:** RAG (Retrieval Augmented Generation), CoT (Chain of Thought).
* **Implementation:** `basicLLM` (A basic implementation of transformer-based concepts).

### Part 4: Security & Risks (Critical)
* **The Black Box Problem:** Interpretability crisis.
* **Attacks:** Prompt Injection, Jailbreaking, Data Poisoning.
* **Defense:** OWASP Top 10 for LLM Applications 2025.

---

## Running Example Models

### basicLLM

#### Prerequisites
* Python 3.8+
* Virtualenv (recommended)

#### Installation & Running `basicLLM`

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/samliumay/AI-Beyond-the-Hype-Architecture-Reality-and-Risks.git](https://github.com/samliumay/AI-Beyond-the-Hype-Architecture-Reality-and-Risks.git)
    cd AI-Beyond-the-Hype-Architecture-Reality-and-Risks
    ```

2.  **Navigate to the example folder:**
    ```bash
    cd basicLLM
    ```

3.  **Activate Python virtual environment:**
    * *Windows:*
        ```powershell
        .\.venv\Scripts\Activate.ps1
        ```
    * *Linux/Mac:*
        ```bash
        source .venv/bin/activate
        ```

4.  **Run the model:**
    ```bash
    python run_model.py
    ```

---

## Roadmap & Future Plans

* Detailed Explanation about how LLM (Transformer Based) training works integrated to Slides.

* LLM training code (Transformer Based) by using torch, numpy and tokenizer.

* Detailed Explanation about RL and how it works integrated to Slides.

* Different RL model will be written as a service like BasicLLM.

* Differen Pure RL training codes will be provided for different models.

---

## 🔗 References & Inspiration

* **Learning:** [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
* **Security:** [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
* **Threat Matrix:** [MITRE ATLAS](https://atlas.mitre.org/matrices/ATLAS)
* **Community:** Special thanks to @[malibayram](https://github.com/malibayram) for insights on LLM training scripts.

---

## 👨‍💻 Author

**Umay SAMLI**
* Connect regarding AI Architecture & Security.
* [GitHub Profile](https://github.com/samliumay)
* [Email](mailto:samliumay965@gmail.com)

---
*Started as a knowledge-sharing initiative at @infinitumIT.*
