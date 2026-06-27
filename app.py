import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Next Word Predictor",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Background */
.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b);
font-family:'Segoe UI',sans-serif;
}

/* Main Title */
.main-title{
text-align:center;
font-size:48px;
font-weight:800;
color:white;
margin-top:20px;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#cbd5e1;
margin-bottom:35px;
}

/* Card */
.card{
background:rgba(255,255,255,.08);
padding:28px;
border-radius:20px;
backdrop-filter:blur(15px);
border:1px solid rgba(255,255,255,.15);
box-shadow:0px 8px 30px rgba(0,0,0,.30);
}

/* Input */
.stTextInput input{
background:#111827 !important;
color:white !important;
border-radius:12px !important;
border:2px solid #3b82f6 !important;
font-size:18px;
height:55px;
}

/* Buttons */
.stButton>button{
width:100%;
height:52px;
border:none;
border-radius:12px;
font-size:18px;
font-weight:700;
background:linear-gradient(90deg,#2563eb,#3b82f6);
color:white;
transition:0.3s;
}

.stButton>button:hover{
transform:translateY(-2px);
box-shadow:0px 8px 20px rgba(37,99,235,.4);
}

/* Result Card */
.result-card{
background:#111827;
border-left:6px solid #22c55e;
border-radius:18px;
padding:25px;
margin-top:25px;
text-align:center;
box-shadow:0px 5px 15px rgba(0,0,0,.3);
}

.result-title{
font-size:20px;
color:#cbd5e1;
}

.result-word{
font-size:40px;
font-weight:bold;
color:#22c55e;
margin-top:10px;
}

/* Footer */
.footer{
text-align:center;
color:#94a3b8;
margin-top:35px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_resources():

    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl","rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl","rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------
def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len-1,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    predicted_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break

    return predicted_word, confidence, prediction
# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
"""
<div class="main-title">
🤖 AI Next Word Predictor
</div>

<div class="subtitle">
Deep Learning • TensorFlow • LSTM • Streamlit
</div>
""",
unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.title("📊 Model Information")

    st.success("✅ Model Loaded Successfully")

    st.markdown("---")

    st.metric("Model", "LSTM")

    st.metric("Vocabulary", len(tokenizer.word_index))

    st.metric("Sequence Length", max_len)

    st.markdown("---")

    st.write("### 💡 Example Inputs")

    st.code("Deep learning is")

    st.code("Artificial intelligence")

    st.code("Machine learning")

    st.code("Natural language")

# ---------------------------------------------------
# MAIN CARD
# ---------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("💬 Enter a sentence")

user_input = st.text_input(
    "",
    placeholder="Example : Deep learning is"
)

col1, col2 = st.columns(2)

with col1:

    predict_btn = st.button(
        "🚀 Predict",
        use_container_width=True
    )

with col2:

    clear_btn = st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear_btn:
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

prediction_box = st.empty()
# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_btn:

    if user_input.strip() == "":
        st.warning("⚠ Please enter a sentence.")
    else:

        with st.spinner("🤖 AI is predicting..."):

            next_word, confidence, prediction = predict_next_word(user_input)

        # Main Result
        prediction_box.markdown(f"""
        <div class="result-card">

        <div class="result-title">
        🤖 Predicted Next Word
        </div>

        <div class="result-word">
        {next_word}
        </div>

        <br>

        <b style="color:white;">
        Confidence : {confidence:.2f}%
        </b>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 📈 Top 5 Predictions")

        probabilities = prediction[0]

        top5 = np.argsort(probabilities)[-5:][::-1]

        index_word = {v: k for k, v in tokenizer.word_index.items()}

        for idx in top5:

            word = index_word.get(idx, "Unknown")

            prob = probabilities[idx] * 100

            c1, c2 = st.columns([2, 5])

            with c1:
                st.write(f"**{word}**")

            with c2:
                st.progress(float(prob / 100))
                st.caption(f"{prob:.2f}%")

        st.markdown("---")

        st.subheader("📊 Prediction Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Prediction", next_word)

        with c2:
            st.metric("Confidence", f"{confidence:.2f}%")

        with c3:
            st.metric("Vocabulary", len(tokenizer.word_index))

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("""
<div class="developer">
    <h3>👨‍💻 Made by</h3>
    <h1>VIKAS KUMAR</h1>
</div>
""", unsafe_allow_html=True)