# Predicting-Human-Eye-Diseases-Using-deep-Learning-on-OCT-Images

This repository contains the code and report for the MSc Data Science project

## Project Overview
**Objective**: Compare lightweight deep learning models (MobileNetV3 Large and EfficientNetB0) for classifying retinal OCT images into four classes:
**Choroidal Neovascularization (CNV)**, **Diabetic Macular Edema (DME)**, **Drusen**, and **Normal**.  
**Key Contributions**:
- Achieved **~98.8% test accuracy** with optimized computational efficiency.
- Deployed a Streamlit web app for real-time clinical predictions.
- Addressed data leakage, and ethical biases.

## Key Features
- **Lightweight Models**: MobileNetV3 (22 ms/inference, 11.9 MB) and EfficientNetB0 (35 ms/inference, 16.08 MB).
- **High Diagnostic Performance**:
  - MobileNetV3: 100% precision for Normal cases.
  - EfficientNetB0: 100% recall for urgent CNV detection.
- **Streamlit Deployment**: Web app with confidence thresholds and clinical context.

## Dataset
**Retinal OCT Images (Kermany et al., 2018)**  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018/data) 
- **Classes**: CNV, DME, Drusen, Normal (84,495 images initially).  
- **Preprocessing**: Deduplication, stratified sampling, and augmentation (rotation, zoom, contrast).

## Installation
1. **Clone the repository**:
   git clone https://github.com/Rakesh939297/Predicting-Human-Eye-Diseases-Using-Machine-Learning-on-OCT-Images.git
   cd Predicting-Human-Eye-Diseases-Using-Machine-Learning-on-OCT-Images
   
3. **Install dependencies**:
  pip install -r requirements.txt

  **Requirements:**
  tensorflow scikit-learn numpy matplotlib seaborn pandas streamlit


**Run the web app:**
streamlit run website.py

**Results:**

____________________________________________________________________________
| Metric               | MobileNetV3 Large       | EfficientNetB0          |
|----------------------|-------------------------|-------------------------|
| **Accuracy**         | 98.86%                  | 98.76%                  |
| **Inference Time**   | 22 ms/image             | 35 ms/image             |
| **Model Size**       | 11.9 MB                 | 16.08 MB                |
| **CNV Recall**       | 98.3%                   | 100%                    |
| **DME Precision**    | 100%                    | 100%                    |
| **Drusen F1-Score**  | 98%                     | 98%                     |
| **AUC-ROC (CNV)**    | 0.9994                  | 0.9998                  |
| **AUC-ROC (Macro)**  | 0.9993                  | 0.9990                  |
____________________________________________________________________________




