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