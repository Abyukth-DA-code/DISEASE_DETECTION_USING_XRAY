# DCPL: Deep Learning for Multi-Class Pulmonary Disease Identification from Chest X-rays

A deep learning-based medical image classification project that uses Convolutional Neural Networks (CNNs) to classify chest X-ray images into four categories: **COVID-19, Pneumonia, Lung Cancer, and Normal**. The framework also incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize image regions influencing the model's predictions.

## Overview

Pulmonary diseases require timely identification to support clinical assessment and treatment planning. This project explores automated chest X-ray classification using deep learning, along with explainable AI techniques to make model predictions more interpretable.

The DCPL (Deep Convolutional Pulmonary Learning) framework combines image preprocessing, CNN-based feature extraction, multi-class classification, model evaluation, and Grad-CAM visualization.

## Key Features

- **Four-Class Classification:** Identifies COVID-19, pneumonia, lung cancer, and normal chest X-ray images.
- **Deep Learning:** Uses a Sequential CNN architecture to learn relevant visual features.
- **Image Preprocessing:** Resizes images to 64 × 64 pixels and normalizes pixel values.
- **Data Augmentation:** Uses techniques such as rotation, zoom, and horizontal flipping to improve generalization.
- **Explainable AI:** Generates Grad-CAM heatmaps to visualize regions influencing predictions.
- **Model Evaluation:** Uses accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC analysis.
- **Model Comparison:** Explores comparisons with established CNN architectures, including VGG16.

## System Workflow

1. Collect chest X-ray images from publicly available datasets.
2. Preprocess images through resizing and normalization.
3. Apply data augmentation to improve model robustness.
4. Train the CNN model for four-class classification.
5. Predict the class of an input chest X-ray.
6. Evaluate model performance using classification metrics.
7. Generate Grad-CAM visualizations to inspect prediction-relevant regions.

## Model Architecture

The DCPL implementation uses a Sequential CNN with the following components:

- Convolutional layers for visual feature extraction.
- Max-pooling layers for spatial downsampling.
- A flattening layer to convert feature maps into a vector.
- A fully connected dense layer with 256 neurons.
- A four-unit output layer for class prediction.

**Training configuration**
- Optimizer: Adam
- Loss function: Categorical cross-entropy
- Input image size: 64 × 64 pixels
- Batch size: 16, as described in the research methodology
- Output: Four-class prediction

## Dataset

The research methodology discusses publicly available chest imaging datasets, including:

- [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
- [ChestX-ray8](https://nihcc.app.box.com/v/ChestXray-Nihcc)
- [COVID-19 Image Data Collection](https://github.com/ieee8023/covid-chestxray-dataset)
- [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- [COVID-QU-Ex Dataset](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)
- [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

The implementation notebook also demonstrates evaluation using a local sample dataset containing 50 test images across four classes.

**Note:** Download datasets separately and follow their respective licenses and access requirements. The research methodology and the notebook's available sample data represent different dataset scopes.

## Results

### Research Paper Results

The research paper reports the following results for the proposed DCPL model:

| Metric | Reported result |
|---|---:|
| Accuracy | 94.8% |
| Precision | 95.2% |
| Recall | 94.5% |
| F1-score | 94.8% |

The paper also presents confusion matrix analysis, ROC curves, and Grad-CAM visualizations.

### Notebook Evaluation

In the provided notebook, the saved DCPL model achieved **88% test accuracy** on a 50-image test set, with a test loss of 0.4924.

These results are from different evaluation contexts and should not be treated as directly comparable without reproducing the same dataset, preprocessing, model checkpoint, and evaluation protocol.

## Explainable AI with Grad-CAM

Grad-CAM helps visualize the image regions that contribute to a CNN's class prediction by generating a heatmap from gradients and convolutional feature maps.

The visualization workflow:

1. Predicts a class for an input chest X-ray.
2. Identifies the final convolutional layer.
3. Computes gradients for the predicted class.
4. Generates and normalizes a class-activation heatmap.
5. Overlays the heatmap on the original X-ray.

This provides a visual aid for examining model behavior. It does not establish that a prediction is clinically correct or that the highlighted regions are medically valid.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Seaborn
- Convolutional Neural Networks (CNNs)
- Grad-CAM and Explainable AI

## Project Structure

The repository includes the following implementation components:

```text
DCPL/
├── app.py
├── main.ipynb
├── Model.ipynb
├── DCPL.h5
├── requirement.txt
├── templates/
├── static/
├── uploads/
└── Sample_dataset/
    └── Test/
        ├── Lung_Cancer/
        ├── PNEUMONIA/
        ├── covid/
        └── normal/
```

*The structure above reflects the files and directories identified in the supplied project notebook. Your actual GitHub repository may differ.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

If your repository contains `requirement.txt`, run:

```bash
pip install -r requirement.txt
```

If the dependency file is named `requirements.txt`, use that filename instead.

### 4. Run the application

If `app.py` contains the configured web application entry point, start it using:

```bash
python app.py
```

Follow the terminal output to access the application. Ensure that the required model file and dependencies are available.

### 5. Run the notebook

Open `main.ipynb` or `Model.ipynb` in Jupyter Notebook or Google Colab to explore model loading, prediction, evaluation, and Grad-CAM visualization.

Update file paths to match your local environment before executing the cells.

## Future Improvements

- Expand and diversify the training and testing datasets.
- Validate performance using independent, patient-level test sets.
- Improve class-wise sensitivity and robustness.
- Explore transfer learning and fine-tuning of pretrained CNN architectures.
- Integrate clinical metadata when suitable validated data is available.
- Extend explainability with additional methods such as SHAP or Layer-wise Relevance Propagation.
- Improve deployment efficiency and reproducibility.

## Disclaimer

This project is intended for **educational and research purposes only**. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical advice, radiologist interpretation, or clinical decision-making.

## Acknowledgements

We acknowledge Dayananda Sagar University and the publicly available datasets and open-source tools that supported this research.

## Authors

**DA Abyukth** and project collaborators  
Department of Computer Science and Engineering (AI & ML)  
Dayananda Sagar University, Bengaluru, India

---

*DCPL — Exploring multi-class pulmonary disease classification through deep learning and explainable AI.*
