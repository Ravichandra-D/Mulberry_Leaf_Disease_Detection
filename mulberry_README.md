# 🌿 Mulberry Leaf Disease Detection using Deep Learning

## 📌 Project Overview

This project is an AI-powered web application that detects diseases in
mulberry leaves using a deep learning model.

Users can upload an image of a mulberry leaf, and the system predicts
whether the leaf is Healthy, Leaf Spot, Powdery Mildew, or Fertilizer
Burn.

The application is deployed using Streamlit Cloud and uses a trained
TensorFlow model for predictions.

------------------------------------------------------------------------

## 🌐 Live Demo

👉 https://mulberryleafdiseasedetection-xxxx.streamlit.app

------------------------------------------------------------------------

## 🎯 Features

-   Upload leaf image\
-   AI-based disease detection\
-   Confidence score display\
-   Probability graph visualization\
-   Interactive UI using Streamlit

------------------------------------------------------------------------

## 🧠 Model Details

-   Deep learning image classification model\
-   Input size: 224 × 224\
-   Output classes:
    -   Healthy\
    -   Leaf Spot\
    -   Powdery Mildew\
    -   Fertilizer Burn

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python\
-   TensorFlow / Keras\
-   Streamlit\
-   NumPy\
-   Matplotlib\
-   Pillow (PIL)

------------------------------------------------------------------------

## 📂 Project Structure

Mulberry_Leaf_Disease_Detection/

├── app.py\
├── requirements.txt\
├── runtime.txt\
├── mulberry_saved_model/\
│ ├── saved_model.pb\
│ ├── fingerprint.pb\
│ └── variables/\
└── README.md

------------------------------------------------------------------------

## 🚀 How to Run Locally

1.  Clone the repository\
    git clone
    https://github.com/your-username/Mulberry_Leaf_Disease_Detection.git

2.  Navigate to folder\
    cd Mulberry_Leaf_Disease_Detection

3.  Install dependencies\
    pip install -r requirements.txt

4.  Run app\
    streamlit run app.py

------------------------------------------------------------------------

## ☁️ Deployment (Streamlit Cloud)

1.  Push code to GitHub\
2.  Go to https://share.streamlit.io\
3.  Connect repo\
4.  Select app.py\
5.  Click Deploy

------------------------------------------------------------------------

## 📊 Workflow

1.  Upload image\
2.  Image preprocessing\
3.  Model prediction\
4.  Display result

------------------------------------------------------------------------

## 🌱 Future Improvements

-   Add more diseases\
-   Improve accuracy\
-   Add treatment suggestions\
-   Mobile-friendly UI

------------------------------------------------------------------------

## 👨‍💻 Author

Ravichandra D
