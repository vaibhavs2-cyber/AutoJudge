import streamlit as st
import joblib
import re
import pandas as pd
st.set_page_config(page_title="AutoJudge AI", page_icon="⚖️")
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
@st.cache_resource #using it here so it doesn't reload the models everytime when a button is clicked.....
def load_models():
    clf_model = joblib.load('difficulty_model.pkl') #(Difficulty Label)
    clf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    reg_model = joblib.load('score_model.pkl') #(Score)
    reg_vectorizer = joblib.load('tfidf_vectorizer_reg.pkl')
    return clf_model, clf_vectorizer, reg_model, reg_vectorizer
try: # Now we'll Load .....
    clf_model, clf_vectorizer, reg_model, reg_vectorizer = load_models()
    st.success("✅ AI Systems Online")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()
st.title("🧩 Vaibhav's Algo-Meter") # Now, I'm the designing the User Interface...
with st.sidebar:
    st.write("## About")
    st.write("Built by Vaibhav Sharma")
    st.write("Project for ACM IITR")
st.markdown("Paste your programming problem details below to predict its difficulty.")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Problem Title", placeholder="e.g., Two Sum")
    with col2:
        st.text_input("Source/Platform", placeholder="e.g., Codeforces") # did this for just for some display......
    description = st.text_area("Problem Description", height=150, placeholder="Write a program that...")
    col3, col4 = st.columns(2)
    with col3:
        input_desc = st.text_area("Input Description", height=100, placeholder="The first line contains an integer N...")
    with col4:
        output_desc = st.text_area("Output Description", height=100, placeholder="Print the sum of...")

if st.button("🔮 Predict Difficulty", type="primary"): # The Prediction Logic....
    if not description:
        st.warning("⚠️ Please at least enter a Problem Description.")
    else:
        raw_text = f"{title} {description} {input_desc} {output_desc}" # combining the inputs exactly like I did above...
        clean_input = clean_text(raw_text)
        # Predicting the Class (Easy/Medium/Hard)
        vec_input_clf = clf_vectorizer.transform([clean_input]) 
        prediction_class_idx = clf_model.predict(vec_input_clf)[0]
        class_map = {0: "Easy", 1: "Medium", 2: "Hard"} # I have mapped 0,1,2 back to names.....
        predicted_label = class_map.get(prediction_class_idx, "Unknown")
        # Predicting the Score (Number)
        vec_input_reg = reg_vectorizer.transform([clean_input])
        predicted_score = reg_model.predict(vec_input_reg)[0]
        st.divider() # Now finally displaying the results....
        st.subheader("📊 Analysis Results")
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.metric(label="Predicted Difficulty", value=predicted_label)
            if predicted_label == "Easy":
                st.success("Level: Beginner Friendly")
            elif predicted_label == "Medium":
                st.warning("Level: Intermediate")
            else:
                st.error("Level: Advanced / Expert")
        with r_col2:
            st.metric(label="Complexity Score (1-10)", value=f"{predicted_score:.2f}")
            st.progress(min(predicted_score / 10, 1.0)) # Observing a nice progress bar visual....