import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mulberry Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp{
background: linear-gradient(-45deg,#d4fc79,#96e6a1,#c3f3d6,#e9f9ef);
background-size: 400% 400%;
animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG{
0%{background-position:0% 50%}
50%{background-position:100% 50%}
100%{background-position:0% 50%}
}
.big-title{
font-size:70px !important;
font-weight:900 !important;
text-align:center;
background: linear-gradient(90deg,#1b5e20,#4caf50,#2e7d32);
-webkit-background-clip:text;
color:transparent;
margin-bottom:5px;
}
.subtitle{
font-size:30px !important;
text-align:center;
color:#1b5e20;
margin-bottom:40px;
}
.card{
background: rgba(255,255,255,0.6);
backdrop-filter: blur(10px);
padding:30px;
border-radius:20px;
box-shadow:0px 10px 25px rgba(0,0,0,0.1);
}
.result-box{
background:#f1f8e9;
padding:25px;
border-radius:15px;
border-left:8px solid #4caf50;
font-size:22px;
font-weight:600;
margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 Mulberry AI")
st.sidebar.info("""
### About
AI system that detects diseases in **Mulberry leaves**.

### Detectable Diseases
✔ Healthy  
✔ Leaf Spot  
✔ Powdery Mildew  
✔ Fertilizer Burn
""")

# ---------------- TITLE ----------------
st.markdown('<div class="big-title">🌿 Mulberry Leaf Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">📊 AI Powered Mulberry Disease Detection System</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
from keras.layers import TFSMLayer

@st.cache_resource
def load_my_model():
    model = TFSMLayer(
        "mulberry_saved_model",
        call_endpoint="serving_default"
    )
    return model

model = load_my_model()

classes = ['Fertilizer', 'Healthy', 'LeafSpot', 'Powdery']

# ---------------- NEW: DISEASE INFO ----------------
disease_info = {

"Healthy": {
"description": "The mulberry leaf is healthy with no visible disease symptoms.",
"treatment": """
• Maintain proper watering and sunlight  
• Apply balanced fertilizer regularly  
• Monitor plants periodically  
• Remove weeds around plants  
"""
},

"LeafSpot": {
"description": "Leaf Spot is a fungal disease causing brown/black spots.",
"treatment": """
• Remove infected leaves  
• Spray copper fungicide or mancozeb  
• Avoid overwatering  
• Maintain plant spacing  
"""
},

"Powdery": {
"description": "Powdery Mildew appears as white powder on leaves.",
"treatment": """
• Use sulfur fungicide or neem oil  
• Remove infected leaves  
• Improve airflow  
• Avoid excess nitrogen fertilizer  
"""
},

"Fertilizer": {
"description": "Fertilizer burn due to excess nutrients.",
"treatment": """
• Reduce fertilizer usage  
• Flush soil with water  
• Follow proper dosage  
• Prefer organic compost  
"""
}
}

# ---------------- INPUT ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

uploaded_file = None
camera_image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","jpeg","png"])
else:
    camera_image = st.camera_input("📸 Take Photo")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if model is not None and (uploaded_file is not None or camera_image is not None):

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = Image.open(camera_image).convert("RGB")

    st.image(image, caption="🌿 Input Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model(img_array)

    if isinstance(prediction, dict):
        prediction = list(prediction.values())[0]

    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()

    prediction = np.array(prediction)

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    st.success("Prediction completed successfully!")

    # -------- RESULT --------
    st.markdown(f"""
    <div class="result-box">
    🌱 Predicted Disease : <b>{predicted_class}</b><br>
    📊 Confidence : <b>{confidence*100:.2f}%</b>
    </div>
    """, unsafe_allow_html=True)

    # -------- NEW: DESCRIPTION --------
    st.subheader("🌿 Disease Description")
    st.write(disease_info[predicted_class]["description"])

    # -------- NEW: TREATMENT --------
    st.subheader("💊 Treatment Recommendation")
    st.success(disease_info[predicted_class]["treatment"])

    # -------- CONFIDENCE BAR --------
    st.subheader("📊 Model Confidence")
    st.progress(confidence)

    # -------- CHART --------
    fig, ax = plt.subplots()
    colors = ['#ef5350','#66bb6a','#ffa726','#42a5f5']
    ax.bar(classes, prediction[0], color=colors)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Distribution")
    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>
🌿 Mulberry Disease Detection System
</center>
""", unsafe_allow_html=True)
