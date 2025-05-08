import streamlit as st
import os
import fitz  # PyMuPDF
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# === Setup ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# === Load default article ===
DEFAULT_PDF_PATH = "Retrieval-Augmented Generation (RAG) from basics to advanced _ by Tejpal Kumawat _ Medium.pdf"

@st.cache_resource
def load_default_article():
    with open(DEFAULT_PDF_PATH, "rb") as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    vs = FAISS.from_documents(docs, OpenAIEmbeddings(model="text-embedding-ada-002"))
    return text, vs

# ✅ New: Function to display the PDF as a scrollable viewer
def display_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" height="800px" 
            style="border: none;"
            type="application/pdf">
        </iframe>
        """
        st.components.v1.html(pdf_display, height=800)
    except Exception as e:
        st.error("❌ Could not display the PDF.")
        st.code(str(e))

# === Load and cache default article if not already
if "default_text" not in st.session_state:
    st.session_state["default_text"], st.session_state["default_vs"] = load_default_article()
    st.session_state["default_chain"] = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        retriever=st.session_state["default_vs"].as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

# ✅ Track which tab to display on rerun
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "🏠 Home"

# === Tabs with dynamic tab switch
tabs = ["🏠 Home", "💬 Chatbot", "📄 Default Article", "🎓 Tutorial Q&A"]
tab_index = tabs.index(st.session_state["active_tab"])
tab1, tab2, tab3, tab4 = st.tabs(tabs)

# --- 🏠 Home ---
with tab1:
    st.title("ScholarAI")
    st.markdown("Welcome to your AI-powered research assistant!")

    uploaded_file = st.file_uploader("Upload your own PDF", type="pdf")
    if uploaded_file:
        st.session_state["uploaded_pdf"] = uploaded_file
        for key in ["vectorstore", "conversation_chain", "messages", "memory"]:
            st.session_state.pop(key, None)

    if st.button("Go to Chat"):
        if "uploaded_pdf" in st.session_state:
            st.session_state["active_tab"] = "💬 Chatbot"
            st.rerun()
        else:
            st.warning("Please upload a PDF first.")

# --- 💬 Chatbot ---
# --- 💬 Chatbot ---
with tab2:
    st.title("Ask Your Paper")

    # Case 1: User uploaded a custom PDF
    if "uploaded_pdf" in st.session_state:
        pdf_file = st.session_state["uploaded_pdf"]
        st.write(f"📄 Currently using: `{pdf_file.name}`")

        if "vectorstore" not in st.session_state:
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = "".join([page.get_text() for page in doc])
            docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).create_documents([text])
            st.session_state["vectorstore"] = FAISS.from_documents(docs, OpenAIEmbeddings(model="text-embedding-ada-002"))

        if "conversation_chain" not in st.session_state:
            st.session_state["conversation_chain"] = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
                retriever=st.session_state["vectorstore"].as_retriever(),
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            )

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User input
        user_prompt = st.chat_input("Ask something about your uploaded PDF")
        if user_prompt:
            st.session_state["messages"].append({"role": "user", "content": user_prompt})
            response = st.session_state["conversation_chain"].invoke({"question": user_prompt})
            st.session_state["messages"].append({"role": "assistant", "content": response['answer']})
            st.rerun()

    # Case 2: No uploaded file → use default article
    else:
        st.info("No file uploaded. Chatting with the default tutorial article instead.")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User input
        user_prompt = st.chat_input("Ask something about the default tutorial article")
        if user_prompt:
            st.session_state["messages"].append({"role": "user", "content": user_prompt})
            response = st.session_state["default_chain"].invoke({"question": user_prompt})
            st.session_state["messages"].append({"role": "assistant", "content": response['answer']})
            st.rerun()

# --- 📄 Default Article ---
with tab3:
    st.header("📄 Default Article (Original Format)")
    display_pdf(DEFAULT_PDF_PATH)  # Preview; you can use expander or scrolling text

# --- 🎓 Tutorial Q&A ---
with tab4:
    st.header("🎓 Tutorial Mode: Q&A Based on Default Article")

    sample_questions = [
        "What is Retrieval-Augmented Generation (RAG)?",
        "What is the 'Lost in the Middle' issue?",
        "How does the article propose to improve efficiency?",
    ]

    selected = st.selectbox("Select a tutorial question", sample_questions)
    if st.button("Ask Selected"):
        res = st.session_state["default_chain"].invoke({"question": selected})
        st.markdown(f"**Answer:** {res['answer']}")

    user_q = st.chat_input("Ask your own question about the tutorial article")
    if user_q:
        res = st.session_state["default_chain"].invoke({"question": user_q})
        st.markdown(f"**Bot:** {res['answer']}")
