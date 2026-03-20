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

/* Animated gradient background */
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

/* BIG TITLE */
.big-title{
font-size:70px !important;
font-weight:900 !important;
text-align:center;
background: linear-gradient(90deg,#1b5e20,#4caf50,#2e7d32);
-webkit-background-clip:text;
color:transparent;
margin-bottom:5px;
}

/* Subtitle */
.subtitle{
font-size:30px !important;
text-align:center;
color:#1b5e20;
margin-bottom:40px;
}

/* Glass card */
.card{
background: rgba(255,255,255,0.6);
backdrop-filter: blur(10px);
padding:30px;
border-radius:20px;
box-shadow:0px 10px 25px rgba(0,0,0,0.1);
}

/* Prediction result */
.result-box{
background:#f1f8e9;
padding:25px;
border-radius:15px;
border-left:8px solid #4caf50;
font-size:22px;
font-weight:600;
margin-top:20px;
}

/* Buttons */
.stButton>button{
background:linear-gradient(90deg,#43a047,#1b5e20);
color:white;
border-radius:10px;
font-size:18px;
padding:10px 20px;
border:none;
}

.stButton>button:hover{
background:linear-gradient(90deg,#1b5e20,#43a047);
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 Mulberry AI")
st.sidebar.info(
"""
### About
AI system that detects diseases in **Mulberry leaves**.

### Detectable Diseases
✔ Healthy  
✔ Leaf Spot  
✔ Powdery Mildew  
✔ Fertilizer Burn
"""
)

st.sidebar.success("Upload a leaf image to start detection.")


# ---------------- TITLE ----------------
st.markdown('<div class="big-title">🌿 Mulberry Leaf Disease Classifier</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">📊 AI Powered Mulberry Disease Detection System</div>', unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
#@st.cache_resource
#def load_model():
 #   model = tf.keras.models.load_model("mulberry_best_model.h5", compile=False)
  #  return model

from keras.models import load_model

@st.cache_resource
def load_model():
    model = load_model(
        "mulberry_best_model.h5",
        compile=False,
        safe_mode=False   # ✅ IMPORTANT
    )
    return model
model = load_model()

classes = ['Fertilizer', 'Healthy', 'LeafSpot', 'Powdery']


# ---------------- FILE UPLOAD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Mulberry Leaf Image", type=["jpg","jpeg","png"])

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="🌿 Uploaded Leaf Image", use_container_width=True)

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)


    # -------- RESULT CARD --------
    st.markdown(f"""
    <div class="result-box">
    🌱 Predicted Disease : <b>{predicted_class}</b><br>
    📊 Confidence Score : <b>{confidence*100:.2f}%</b>
    </div>
    """, unsafe_allow_html=True)


    # -------- CONFIDENCE BAR --------
    st.subheader("📊 Model Confidence")

    st.progress(float(confidence))


    # -------- PROBABILITY CHART --------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Disease Prediction Probabilities")

    fig, ax = plt.subplots()

    colors = ['#ef5350','#66bb6a','#ffa726','#42a5f5']

    ax.bar(classes, prediction[0], color=colors)

    ax.set_ylabel("Probability")

    ax.set_title("Model Prediction Distribution")

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<br><br>
<hr>
<center>

🌿 <b>Mulberry Leaf Disease Detection System</b>  
AI Powered Smart Agriculture Project

</center>
""", unsafe_allow_html=True)
