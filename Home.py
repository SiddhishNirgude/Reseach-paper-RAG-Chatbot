import streamlit as st

col1, col2 = st.columns([4, 1])
with col1:
    st.title("ScholarAI")
with col2:
    on = st.toggle("🌗")

if on:
    theme_color = "#2C3E50"
    font_color = "#ECF0F1"
else:
    theme_color = "#ECF0F1"
    font_color = "#2C3E50"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {theme_color};
        color: {font_color};
    }}
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown('#')
st.markdown(f"""
<div style="text-align: center;">
    <h3 style="font-weight: bold; color: {font_color};">Welcome to ScholarAI!</h3>
</div>

<div style="text-align: center; font-size: 18px; color: {font_color};">
    ScholarAI is an AI-powered research assistant designed to help you with your research papers. 
    You can upload a research paper, and ScholarAI will assist you in answering questions, summarizing 
    key points, or analyzing the paper's content. It’s like having a virtual assistant for your academic needs!
</div>
""", unsafe_allow_html=True)

st.markdown("###")
st.markdown("""<div style="font-size: 18px; color: {font_color};">Upload a PDF</div>""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="pdf")

if uploaded_file:
    st.session_state["uploaded_pdf"] = uploaded_file

st.write("")

st.markdown("""
    <style>
    div.stButton > button {
        color: #2C3E50;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

if st.button("Go to Chat 💬"):
    if "uploaded_pdf" in st.session_state:
        st.switch_page("Chatbot.py")
    else:
        st.warning("📄 Please upload a PDF before proceeding.")
