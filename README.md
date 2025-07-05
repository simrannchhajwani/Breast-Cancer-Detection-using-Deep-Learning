# Breast-Cancer-Detection-using-Deep-Learning
Developed a deep learning model for classifying breast cancer, achieving 99.45% accuracy on histopathology images. This project uses a Swin V2 Transformer with CLAHE-enhanced data from the BreaKHis dataset to deliver state-of-the-art diagnostic performance for benign and malignant tissue detection.





# High-Accuracy Breast Cancer Detection using Swin V2 Transformers

This repository contains the official implementation for the paper/project titled: **"Comparative Analysis of Swin Transformer V2 for High-Accuracy Breast Cancer Detection in Histopathological Images."** This work focuses on leveraging state-of-the-art deep learning models to achieve highly accurate binary classification (benign vs. malignant) of breast cancer from histopathological images.

![image](https://github.com/user-attachments/assets/03e19ca0-3242-4eea-8140-c244b5ea4833)



---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Results](#results)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

Breast cancer remains a significant cause of mortality among women worldwide, making early and precise diagnosis critical for improving patient outcomes. This project explores the application of Vision Transformers, specifically the Swin Transformer V2 architecture, for the automated classification of breast cancer histopathological images.

We utilize the **BreaKHis dataset** and enhance it using **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to improve image quality and normalize variations. Our comparative analysis evaluates the performance of Swin V2 variants (Tiny, Small, and Base) across four different magnification levels (40X, 100X, 200X, 400X), demonstrating the model's robustness and superior performance in this critical diagnostic task.

![image](https://github.com/user-attachments/assets/7c38ab01-cac5-4859-b89a-2a4b3e0c09ab)

---

## Problem Statement

The classification of histopathological images is inherently challenging due to:
- **High variability** in tissue morphology and cell structures.
- **Inconsistencies** in staining and image quality across slides.
- **The need for expert pathologists**, whose analysis can be time-consuming and subjective.

Traditional machine learning and even standard CNNs can struggle to capture the fine-grained features and long-range dependencies necessary for accurate diagnosis. This project addresses these challenges by employing the hierarchical and window-based self-attention mechanism of the Swin Transformer.

---

## Key Features

- **State-of-the-Art Model:** Implements the Swin Transformer V2, a powerful architecture for computer vision tasks.
- **High-Accuracy Classification:** Achieves a peak accuracy of **99.45%** in distinguishing between benign and malignant tissues.
- **Multi-Magnification Analysis:** The model is rigorously evaluated on images at 40X, 100X, 200X, and 400X magnifications.
- **Advanced Preprocessing:** Uses CLAHE to enhance image contrast and standardize input data, leading to improved model performance.
- **Comprehensive Evaluation:** Performance is measured using accuracy, precision, recall, F1-score, and Area Under the Curve (AUC).

---

## Methodology

The project workflow is structured as follows:

1.  **Dataset:** The [BreaKHis dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) is used, containing thousands of histopathological images categorized by malignancy and magnification factor.
2.  **Preprocessing:** All images are preprocessed with CLAHE to normalize lighting and enhance local contrast, revealing finer details in the tissue structure.
3.  **Model Selection:** A comparative study is conducted between three variants of the Swin Transformer V2:
    - `Swin-V2-Tiny`
    - `Swin-V2-Small`
    - `Swin-V2-Base`
4.  **Training & Evaluation:** Each model variant is trained and tested on the preprocessed dataset for all four magnification levels. The performance is logged using the key metrics mentioned above.

---

## Results

The Swin V2 Small variant demonstrated the most robust and superior performance across all metrics.

### Key Performance Highlights (Swin V2 Small)
- **Overall Accuracy:** **99.45%**
- **Area Under the Curve (AUC):** **99.95%**
- **Precision:** **99.64%**
- **Recall:** **99.60%**
- **F1-Score:** **99.56%**

A detailed breakdown of results per magnification level can be found in the associated paper or the `results/` directory.

---

## Tech Stack & Dependencies

- Python 3.8+
- PyTorch
- `timm` (PyTorch Image Models)
- scikit-learn
- OpenCV (for CLAHE)
- NumPy
- Matplotlib
- Pandas

---

## Future Work

Our future research will focus on:
- **Multi-Institutional Validation:** Testing the model's generalizability on data sourced from different hospitals and labs to ensure its robustness in real-world scenarios.
- **Clinical Adaptation:** Optimizing the model for efficient deployment in clinical environments, particularly those with limited computational resources (e.g., through model quantization or pruning).

---

## Citation

If you use this code or find our work helpful in your research, please consider citing our paper.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
