# 😷 Face Mask Detection using MobileNetV2
This project implements a deep learning model for face mask detection using the MobileNetV2 architecture. It classifies images into three categories:

- ✅ Wearing a mask correctly  
- ❌ Not wearing a mask  
- ⚠️ Wearing a mask incorrectly

---

## 📦 Dependencies

Create and activate your virtual environment using one of the following methods:

### 🐍 Option 1: Using `conda`

```bash
conda create --name deepenv python=3.9 
conda activate deepenv
pip install -r requirements.txt
```

### 🐍 Option 2: Using `pip` and `venv`

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ Ensure Python 3.8 or higher is installed.

---

## 📚 Common Packages Used

- `TensorFlow` – Deep learning framework 
- `NumPy` – Numerical operations  
- `Matplotlib` – Visualization  
- `Scikit-learn` – Evaluation & metrics  


---

## 🧠 Model Details

- **Architecture**: MobileNetV2
- **Output**: Multi-class classification  
  - `0`: with_mask  
  - `1`: without_mask  
  - `2`: mask_weared_incorrect  

---

## 🗂️ Dataset and Preprocessing

- Dataset Source: 
  - https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- Dataset contains:
  - `.xml` annotation files for bounding boxes  
  - Corresponding image files  

### 🔍 Preprocessing Pipeline

1. **Parse XML Annotations**  
   - Extract `filename`, `label`, and `bbox` (bounding box) coordinates from `.xml` files

2. **Build DataFrame**  
   - Store annotation data into a `pandas` DataFrame with columns: `filename`, `label`, `bbox`

3. **Load & Preprocess Images**  
   For each annotated image:
   - Load the image using `cv2.imread()`  
   - Crop the face using bounding box coordinates  
   - Resize to `224x224`  
   - Convert to array with `img_to_array()`  
   - Normalize with `preprocess_input()`  

4. **Label Encoding**  
   - Map string labels to integer values using:
     ```python
     label_pair = {
         'with_mask': 0,
         'without_mask': 1,
         'mask_weared_incorrect': 2
     }
     ```

5. **Final Output**  
   - `face_images`: NumPy array of preprocessed face images  
   - `face_labels`: NumPy array of encoded labels  

---
👤 Author
 - Joan Suaverdez -https://github.com/jsuaverd
 - Haasith Srinivasa - https://github.com/DankNub901


