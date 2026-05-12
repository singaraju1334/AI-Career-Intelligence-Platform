import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
import PyPDF2

st.set_page_config(page_title="AI Career Intelligence", layout="wide")

# ---------- THEME ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#000,#0a0a0a); color:white;}
h1,h2,h3 {color:#ffb6c1;}
section[data-testid="stSidebar"] {background:#000;}
.block-container {padding-top:2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Career Intelligence Platform")

page = st.sidebar.radio(
"Navigation",
["📊 Dashboard","🎯 Placement","💼 Role Predictor",
 "📄 Resume AI","🔗 LinkedIn Analyzer","📈 Model Analytics"]
)

# ---------- DATASET ----------
data = {
"cgpa":[5,6,7,8,9,7,6,8,9,5,7,8],
"technical":[30,40,50,60,70,55,45,65,75,35,80,85],
"communication":[35,45,55,65,75,60,50,70,80,40,75,85],
"internships":[0,1,1,2,3,1,2,3,4,0,2,3],
"talent_score":[20,30,40,50,60,45,35,55,65,25,70,80],
"placed":[0,0,1,1,1,1,0,1,1,0,1,1],
"role":[0,1,2,3,4,2,1,3,4,0,5,5]
}

df = pd.DataFrame(data)

X = df[["cgpa","technical","communication","internships","talent_score"]]
y_place = df["placed"]
y_role = df["role"]

rf = RandomForestClassifier().fit(X,y_place)
lr = LogisticRegression().fit(X,y_place)
dt = DecisionTreeClassifier().fit(X,y_place)

role_model = RandomForestClassifier().fit(X,y_role)

role_map = {
0:"Manual Tester",
1:"Software Developer",
2:"Web Developer",
3:"Data Analyst",
4:"Data Scientist",
5:"ML Engineer"
}

# ================= DASHBOARD =================
if page == "📊 Dashboard":

    st.header("Career Analytics Overview")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Students",len(df))
    c2.metric("Placement Rate",str(int(df["placed"].mean()*100))+"%")
    c3.metric("Avg Internships",round(df["internships"].mean(),1))
    c4.metric("Avg Talent Score",round(df["talent_score"].mean(),1))

    st.bar_chart(df[["technical","communication","talent_score"]])

# ================= PLACEMENT =================
elif page == "🎯 Placement":

    st.header("AI Placement Prediction")

    cgpa = st.slider("CGPA",0.0,10.0,7.0)
    tech = st.slider("Technical Skill",0,100,60)
    comm = st.slider("Communication",0,100,60)
    interns = st.slider("Internships",0,6,1)
    talent = st.slider("Extra Talent",0,100,50)

    radar_df = pd.DataFrame({
        "Skill":["Technical","Communication","Talent"],
        "Score":[tech,comm,talent]
    })

    radar = px.line_polar(
        radar_df,r="Score",theta="Skill",
        line_close=True,color_discrete_sequence=["#ff69b4"]
    )
    radar.update_traces(fill='toself')
    st.plotly_chart(radar,width='stretch')

    if st.button("Predict Placement"):

        inp = np.array([[cgpa,tech,comm,interns,talent]])
        prob = rf.predict_proba(inp)[0][1]*100

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text':"Placement Probability"},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':"#ff69b4"}}
        ))
        st.plotly_chart(gauge,width='stretch')

# ================= ROLE =================
elif page == "💼 Role Predictor":

    st.header("AI Career Role Recommendation")

    cgpa = st.slider("CGPA",0.0,10.0,7.0,key=1)
    tech = st.slider("Technical",0,100,60,key=2)
    comm = st.slider("Communication",0,100,60,key=3)
    interns = st.slider("Internships",0,6,1,key=4)
    talent = st.slider("Talent Score",0,100,50,key=5)

    if st.button("Predict Role"):

        inp = np.array([[cgpa,tech,comm,interns,talent]])
        role = role_model.predict(inp)[0]

        st.success(f"🎯 Best Career Role: {role_map[role]}")

# ================= RESUME =================
elif page == "📄 Resume AI":

    st.header("AI Resume Analyzer")

    file = st.file_uploader("Upload Resume PDF",type=["pdf"])

    if file:

        text = ""
        reader = PyPDF2.PdfReader(file)

        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += t

        text = text.lower()
        score = 40

        if "python" in text: score+=10
        if "machine learning" in text: score+=15
        if "sql" in text: score+=10
        if "java" in text: score+=10
        if "project" in text: score+=10

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text':"Resume Strength"},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':"#ffb6c1"}}
        ))

        st.plotly_chart(gauge,width='stretch')

# ================= LINKEDIN =================
elif page == "🔗 LinkedIn Analyzer":

    st.header("LinkedIn Profile Analyzer")

    bio = st.text_area("Paste LinkedIn About Section")

    if st.button("Analyze Profile"):

        score = 50
        bio = bio.lower()

        if "ai" in bio: score+=10
        if "machine learning" in bio: score+=10
        if "leadership" in bio: score+=10
        if "project" in bio: score+=10
        if "internship" in bio: score+=10

        st.success(f"LinkedIn Strength Score: {score}")

# ================= MODEL =================
elif page == "📈 Model Analytics":

    st.header("Model Accuracy Comparison")

    rf_acc = accuracy_score(y_place, rf.predict(X))
    lr_acc = accuracy_score(y_place, lr.predict(X))
    dt_acc = accuracy_score(y_place, dt.predict(X))

    acc_df = pd.DataFrame({
        "Model":["RandomForest","Logistic","DecisionTree"],
        "Accuracy":[rf_acc,lr_acc,dt_acc]
    })

    fig = px.bar(
        acc_df,
        x="Model",
        y="Accuracy",
        color_discrete_sequence=["#ff69b4"]
    )

    st.plotly_chart(fig,width='stretch')
