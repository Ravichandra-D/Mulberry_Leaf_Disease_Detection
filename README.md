# 🌿 Mulberry Leaf Disease Detection using Deep Learning

## 📌 Project Overview

This project is an **AI-based web application** that detects diseases in mulberry leaves using a deep learning model.
The system allows users to upload an image of a mulberry leaf, and the trained model predicts whether the leaf is **healthy or affected by a specific disease**.

The application is built using **TensorFlow for model training** and **Streamlit for creating an interactive web interface**.

This project can help farmers and researchers **identify mulberry leaf diseases quickly**, which may improve crop management and productivity.

---

## 🎯 Objectives

* Detect diseases in mulberry leaves using deep learning.
* Provide a simple web interface for uploading leaf images.
* Display prediction results along with confidence scores.
* Assist farmers and agricultural researchers in early disease detection.

---

## 🧠 Model Details

The model was trained using a dataset of mulberry leaf images and can classify leaves into the following categories:

* **Healthy**
* **Leaf Spot**
* **Powdery Mildew**
* **Fertilizer Burn**

The trained model is saved as:

mulberry_best_model.h5

The model processes images by:

1. Resizing the image to **224 × 224 pixels**
2. Normalizing pixel values
3. Predicting the disease class using the trained neural network.

---

## 💻 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Streamlit**
* **NumPy**
* **Matplotlib**
* **Pillow (PIL)**

---

## 📂 Project Structure

```
Mulberry_Leaf_Disease_Detection
│
├── app.py                     # Streamlit web application
├── mulberry_best_model.h5     # Trained deep learning model
├── requirements.txt           # Required Python libraries
└── README.md                  # Project documentation
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/Mulberry_Leaf_Disease_Detection.git
```

### 2️⃣ Navigate to the Project Folder

```
cd Mulberry_Leaf_Disease_Detection
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit Application

```
streamlit run app.py
```

---

## 🌐 Application Features

* Upload mulberry leaf images
* AI-based disease prediction
* Confidence score visualization
* Probability distribution chart
* Simple and interactive user interface

---

## 📊 Example Workflow

1. User uploads a mulberry leaf image.
2. The system preprocesses the image.
3. The trained model predicts the disease category.
4. The application displays the predicted disease and confidence level.

---

## 🌱 Future Improvements

* Add more mulberry disease classes.
* Improve model accuracy with a larger dataset.
* Add disease treatment suggestions.
* Deploy the system for real-time agricultural usage.

---

## 👨‍💻 Author

**Ravichandra D**

This project was developed as part of a learning initiative in **Deep Learning and AI applications in agriculture**.

---
