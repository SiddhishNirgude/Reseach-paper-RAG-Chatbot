import streamlit as st
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# === Setup ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# === Load default article ===
DEFAULT_PDF_PATH = "data/Retrieval-Augmented Generation (RAG) from basics to advanced _ by Tejpal Kumawat _ Medium.pdf"

@st.cache_resource
def load_default_article():
    with open(DEFAULT_PDF_PATH, "rb") as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    vs = FAISS.from_documents(docs, OpenAIEmbeddings(model="text-embedding-ada-002"))
    return text, vs

if "default_text" not in st.session_state:
    st.session_state["default_text"], st.session_state["default_vs"] = load_default_article()
    st.session_state["default_chain"] = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        retriever=st.session_state["default_vs"].as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "💬 Chatbot",
    "📄 Default Article",
    "🎓 Tutorial Q&A"
])

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
            st.experimental_rerun()
        else:
            st.warning("Please upload a PDF first.")

# --- 💬 Chatbot ---
with tab2:
    st.title("Ask Your Paper")
    if "uploaded_pdf" in st.session_state:
        pdf = st.session_state["uploaded_pdf"]

        if "vectorstore" not in st.session_state:
            pdf.seek(0)
            doc = fitz.open(stream=pdf.read(), filetype="pdf")
            full_text = "".join([p.get_text() for p in doc])
            docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).create_documents([full_text])
            st.session_state["vectorstore"] = FAISS.from_documents(docs, OpenAIEmbeddings(model="text-embedding-ada-002"))

        if "conversation_chain" not in st.session_state:
            st.session_state["conversation_chain"] = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
                retriever=st.session_state["vectorstore"].as_retriever(),
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            )

        st.write(f"📄 Currently using: `{pdf.name}`")
        if st.button("Reset Chat"):
            for key in ["vectorstore", "conversation_chain", "messages", "memory"]:
                st.session_state.pop(key, None)
            st.experimental_rerun()

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask your PDF")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            reply = st.session_state["conversation_chain"].invoke({"question": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": reply["answer"]})
    else:
        st.warning("Please upload a PDF from the Home tab.")

# --- 📄 Default Article ---
with tab3:
    st.header("📄 Default Article Content")
    st.markdown(st.session_state["default_text"][:3000])  # Preview; you can use expander or scrolling text

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
