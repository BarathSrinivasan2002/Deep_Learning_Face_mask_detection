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
