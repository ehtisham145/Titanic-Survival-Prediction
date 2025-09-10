# streamlit_titanic_app.py
# Streamlit frontend for Titanic survival prediction
# Put your trained pipeline (joblib) in the same folder named 'model.pkl'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import uuid

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide", initial_sidebar_state="expanded")

# --------------------------
# Simple CSS for navbar/header/footer + dark mode support
# --------------------------
BASE_CSS = """
<style>
:root{
  --bg:#f8f9fa;
  --card:#ffffff;
  --text:#0f172a;
  --muted:#6b7280;
  --accent:#0366d6;
}
body { background: var(--bg); color:var(--text); }
.header{ padding:1rem 2rem; background:linear-gradient(90deg, #0ea5e9, #7c3aed); color:white; border-radius:12px;}
.navbar{ display:flex; gap:1rem; align-items:center; }
.nav-item{ padding:0.45rem 0.9rem; background:rgba(255,255,255,0.08); border-radius:8px; font-weight:600;}
.card{ background:var(--card); padding:1rem; border-radius:12px; box-shadow:0 4px 18px rgba(2,6,23,0.08);}
.footer{ padding:1rem; text-align:center; color:var(--muted); }
.small{ font-size:0.9rem; color:var(--muted); }
.btn{ background:var(--accent); color:white; padding:0.6rem 1rem; border-radius:10px; font-weight:700; }
</style>
"""

DARK_CSS = """
<style>
:root{
  --bg:#0b1220;
  --card:#071023;
  --text:#e6eef8;
  --muted:#9aa7b5;
  --accent:#3b82f6;
}
body { background: var(--bg); color:var(--text); }
.header{ padding:1rem 2rem; background:linear-gradient(90deg, #111827, #0f172a); color:white; border-radius:12px;}
.navbar{ display:flex; gap:1rem; align-items:center; }
.nav-item{ padding:0.45rem 0.9rem; background:rgba(255,255,255,0.03); border-radius:8px; font-weight:600;}
.card{ background:var(--card); padding:1rem; border-radius:12px; box-shadow:0 6px 24px rgba(2,6,23,0.5);}
.footer{ padding:1rem; text-align:center; color:var(--muted); }
.small{ font-size:0.9rem; color:var(--muted); }
.btn{ background:var(--accent); color:white; padding:0.6rem 1rem; border-radius:10px; font-weight:700; }
</style>
"""

# --------------------------
# Session state for dark mode
# --------------------------
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# toggle control in sidebar will change this

def toggle_dark():
    st.session_state['dark_mode'] = not st.session_state['dark_mode']

# apply CSS
st.markdown(DARK_CSS if st.session_state['dark_mode'] else BASE_CSS, unsafe_allow_html=True)

# --------------------------
# Top navbar + header
# --------------------------
with st.container():
    st.markdown("""
    <div class='header'>
      <div class='navbar'>
        <div style='font-size:1.25rem; font-weight:800;'>Titanic Survival Predictor</div>
        <div style='flex:1'></div>
        <div class='nav-item'>Home</div>
        <div class='nav-item'>About</div>
        <div class='nav-item'>Model Info</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.write("\n")

# --------------------------
# Layout: sidebar for inputs, main for description + results
# --------------------------
sidebar = st.sidebar
sidebar.title("Input Controls")
sidebar.write("Adjust model inputs below")

# Dark mode button
sidebar.checkbox("Dark mode", value=st.session_state['dark_mode'], key='dark_mode_cb', on_change=toggle_dark)

# Interactive presets like GPT-5 style quick-presets
sidebar.subheader("Quick presets")
preset = sidebar.radio("Choose a preset:", ["Select...","Typical Male 3rd class","Female 1st class","Child sample"]) 
if preset == "Typical Male 3rd class":
    preset_vals = {
        'Pclass': 3,
        'Sex': 'male',
        'SibSp': 1,
        'Parch': 0,
        'Embarked': 'S',
        'Title': 'Mr',
        'GrpSize': 'Couple',
        'FareCat': '0-10',
        'AgeCat': '16-32'
    }

elif preset == "Female 1st class":
    preset_vals = {
        'Pclass': 1,
        'Sex': 'female',
        'SibSp': 1,
        'Parch': 0,
        'Embarked': 'C',
        'Title': 'Mrs',
        'GrpSize': 'Couple',
        'FareCat': '70-100',
        'AgeCat': '32-48'
    }

elif preset == "Child sample":
    preset_vals = {
        'Pclass': 2,
        'Sex': 'male',
        'SibSp': 0,
        'Parch': 1,
        'Embarked': 'Q',
        'Title': 'Master',
        'GrpSize': 'Small',   # ❌ galat hai (model me "Small" nahi hai)
        # ✔️ sahi categories me se use karo → Single, Couple, Group, Large Group
        'GrpSize': 'Single',
        'FareCat': '10-25',
        'AgeCat': '0-16'
    }

else:
    preset_vals = {}


sidebar.markdown("---")

# Input fields (main features)
st.sidebar.subheader("Feature inputs")
PassengerId = sidebar.number_input("PassengerId", min_value=1, value=uuid.uuid4().int % 100000)
Pclass = sidebar.selectbox("Pclass", options=[1,2,3], index=(preset_vals.get('Pclass',3)-1) if preset_vals.get('Pclass') else 2)
Sex = sidebar.selectbox("Sex", options=['male','female'], index=0 if preset_vals.get('Sex','male')=='male' else 1)
SibSp = sidebar.number_input("SibSp (siblings/spouses)", min_value=0, max_value=10, value=preset_vals.get('SibSp',0))
Parch = sidebar.number_input("Parch (parents/children)", min_value=0, max_value=10, value=preset_vals.get('Parch',0))
Embarked = sidebar.selectbox("Embarked", options=['C','Q','S'], index=['C','Q','S'].index(preset_vals.get('Embarked','S')) if preset_vals.get('Embarked') else 2)
Title = sidebar.selectbox(
    "Title",
    options=['Mr','Mrs','Miss','Master','Rare Title'],
    index=['Mr','Mrs','Miss','Master','Rare Title'].index(
        preset_vals.get('Title','Mr')
    ) if preset_vals.get('Title') else 0
)
GrpSize = sidebar.selectbox("GrpSize", options=['Single','Couple','Groups','Large Group'], index=['Single','Couple','Groups','Large Group'].index(preset_vals.get('GrpSize','Small')) if preset_vals.get('GrpSize') else 1)
FareCat = sidebar.selectbox(
    "FareCat", 
    options=['0-10','10-25','25-40','40-70','70-100','100+'], 
    index=['0-10','10-25','25-40','40-70','70-100','100+'].index(
        preset_vals.get('FareCat','0-10')
    ) if preset_vals.get('FareCat') else 0
)
AgeCat = sidebar.selectbox(
    "AgeCat", 
    options=['0-16','16-32','32-48','48-64','64+'], 
    index=['0-16','16-32','32-48','48-64','64+'].index(
        preset_vals.get('AgeCat','0-16')
    ) if preset_vals.get('AgeCat') else 0
)

sidebar.markdown("---")

# Upload model button + info
sidebar.subheader("Model")
model_file = sidebar.file_uploader("Upload pipeline (joblib .pkl)", type=['pkl','joblib'])

# --------------------------
# Main column: intro + form + results
# --------------------------
st.markdown("""
<div class='card'>
  <h2>About this app</h2>
  <p class='small'>This is a Streamlit frontend that accepts Titanic passenger features and outputs
  a survival prediction using a pre-trained pipeline. The pipeline should include both preprocessing
  (ColumnTransformer/encoders) and the trained classifier. Upload your pipeline (.pkl) from training.
  </p>
</div>
""", unsafe_allow_html=True)

st.write("\n")

col1, col2 = st.columns((2,1))
with col1:
    st.markdown("### Provide inputs and click Predict")
    st.markdown("**Input summary (no column names):**")
    # show as numpy-array-like without column names
    input_df = pd.DataFrame([{
        'PassengerId': PassengerId,
        'Pclass': Pclass,
        'Sex': Sex,
        'SibSp': SibSp,
        'Parch': Parch,
        'Embarked': Embarked,
        'Title': Title,
        'GrpSize': GrpSize,
        'FareCat': FareCat,
        'AgeCat': AgeCat
    }])
    st.code(input_df.to_numpy().tolist())

    st.markdown("---")
    predict_btn = st.button("Predict")

with col2:
    st.markdown("### Model Info")
    st.write("This model which will predict the survival of titanic passenger on the based of input data which you provide.")

# --------------------------
# load model
# --------------------------
model = None
if model_file is not None:
    # save to disk temporarily and load
    tmp_path = "Tree_clf.pkl"
    with open(tmp_path, "wb") as f:
        f.write(model_file.getbuffer())
    try:
        model = joblib.load(tmp_path)
        st.success("Model uploaded and loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load uploaded model: {e}")

else:
    try:
        model = joblib.load("Tree_clf.pkl")
        st.info("Loaded your Model from app folder.")
    except Exception:
        st.warning("No model found. Please upload a trained pipeline (.pkl) in the sidebar.")

# --------------------------
# Predict when button clicked
# --------------------------
if predict_btn:
    if model is None:
        st.error("No model loaded. Upload or place 'model.pkl' in the app folder.")
    else:
        try:
            # ensure DataFrame columns exactly match training
            # you can enforce dtype conversion if needed
            input_df = input_df.astype({
                'PassengerId': 'int64', 'Pclass':'int64','Sex':'object','SibSp':'int64','Parch':'int64',
                'Embarked':'object','Title':'object','GrpSize':'object','FareCat':'object','AgeCat':'object'
            })

            pred = model.predict(input_df)
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[:,1]

            st.success(f"Predicted class: {pred[0]}")
            if proba is not None:
                st.info(f"Survival probability: {proba[0]:.3f}")

            # show transformed features (if pipeline exposes named steps)
            if hasattr(model, 'named_steps') and 'preprocess' in model.named_steps:
                try:
                    transformed = model.named_steps['preprocess'].transform(input_df)
                    st.write('Transformed array preview:')
                    st.write(np.asarray(transformed))
                except Exception:
                    pass

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --------------------------
# Footer
# --------------------------
st.markdown("\n")
st.markdown("""
<div class='footer'>
  Built with ❤️ using Streamlit — Titanic Survival Predictor · Make sure your pipeline contains the same feature names and preprocessing used during training.
</div>
""", unsafe_allow_html=True)

# --------------------------
# End of file
# --------------------------
