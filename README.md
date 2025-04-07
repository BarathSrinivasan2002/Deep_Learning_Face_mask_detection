# 😷 Face Mask Detection using Deep Learning

This project implements deep learning models to detect face mask usage using Convolutional Neural Networks (CNN), Autoencoders, and the **MobileNetV2** architecture. It classifies faces into three categories:

- ✅ **Wearing a mask correctly**  
- ❌ **Not wearing a mask**  
- ⚠️ **Wearing a mask incorrectly**

---

## 🚀 Getting Started

Follow these steps to get started with the project:

### 1. Run in Google Colab  
Click below to open in Colab:  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](your_colab_notebook_link_here)

### 2. Download the Dataset  
Download the Face Mask Detection dataset from Kaggle:  
🔗 https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

### 3. Install Required Libraries  
Make sure to install the required dependencies:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn
```

---

## 🧼 Preprocessing Pipeline

The following steps outline the preprocessing workflow before training:

### 1. Parse XML Annotations  
Extract the following from `.xml` files:
- `filename`
- `label` (e.g., with_mask, without_mask, mask_weared_incorrect)
- `bbox` (bounding box coordinates)

### 2. Build a DataFrame  
Organize extracted annotation data in a pandas DataFrame with columns:
- `filename`
- `label`
- `bbox`

### 3. Load & Preprocess Images  
For each image:
- Load using `cv2.imread()`
- Crop the face using bounding box values
- Resize to `224x224`
- Convert to array with `img_to_array()`
- Normalize using `preprocess_input()`

### 4. Label Encoding  
Convert string labels to numerical format:

```python
label_pair = {
    'with_mask': 0,
    'without_mask': 1,
    'mask_weared_incorrect': 2
}
```

### 5. Final Output  
- `face_images`: NumPy array of preprocessed face images  
- `face_labels`: NumPy array of encoded labels  

---

## 🧠 Models Used

- ✅ **CNN**  
- 🌀 **Autoencoders**  
- 🏎️ **MobileNetV2** (Transfer Learning)

---

## 📸 Sample Outputs *(Optional Section)*

You can add training accuracy/loss plots and sample prediction results here.

---

## 👨‍💻 Authors

| Name                    | GitHub |
|-------------------------|--------|
| Barath Srinivasan       | [@BarathSrinivasan2002](https://github.com/BarathSrinivasan2002) |
| Muhammad Shahzaib Vohra| [@shahzaibvohra1](https://github.com/shahzaibvohra1) |
| Matias Franco Yion      | [@Matf021](https://github.com/Matf021) |
| Amirmehrbod Panjnoush   | [@amp1414](https://github.com/amp1414) |
| Haasith Srinivasa       | [@DankNub901](https://github.com/DankNub901) |
| Joan Suaverdez          | [@jsuaverd](https://github.com/jsuaverd) |

---



# People and Background Segmentation using SAM (Segment Anything Model)

This project implements a robust pipeline to segment people from the background in face-mask detection images using Meta's **Segment Anything Model (SAM)**. It runs seamlessly in Google Colab and supports KaggleHub integration for dataset downloading.

---

## 📦 Features

- Automatic environment setup and package installation in Colab
- Dataset download via KaggleHub (`andrewmvd/face-mask-detection`)
- Annotation parsing from Pascal VOC-style XML
- Bounding box visualization and expansion
- Integration with Meta's Segment Anything Model (SAM)
- People and background segmentation via box prompts
- Visualization of segmentation masks
- Optimized memory management for Colab

---

## 🚀 Getting Started

### 1. **Open in Google Colab**

> [Run on Colab](https://colab.research.google.com)

> 📁 You must have a Google Drive connected and Kaggle API token ready if downloading from KaggleHub.

---

### 2. **Dataset Access**

Ensure your environment has access to the **Kaggle API**:

- Upload your `kaggle.json` file to Colab, or
- Authenticate through `kagglehub.dataset_download("andrewmvd/face-mask-detection")`

Alternatively, upload the dataset manually and point to the correct folder.

---

## 🧠 Model

This project uses:

- **Model**: `sam_vit_b` (lightweight ViT from Segment Anything)
- **Input**: RGB image and bounding box
- **Output**: Binary segmentation masks for person and background

SAM is loaded using:
```python
from segment_anything import sam_model_registry, SamPredictor
