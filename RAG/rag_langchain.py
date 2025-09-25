import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv("api_key.env", override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API_KEY)

# -------- PDF Reading and Chunking --------
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        reader = PdfReader(pdf)            # UploadedFile can be passed directly
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# -------- Vector Store --------
def vector_store(text_chunks):
    vs = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vs.save_local("faiss_db")

def check_database_exists():
    """Check if FAISS database exists"""
    return os.path.isdir("faiss_db")

# -------- Agent and Response --------
def get_conversational_chain(tool, ques: str):
    # DeepSeek chat model (LangChain init_chat_model)
    llm = init_chat_model("deepseek-chat", model_provider="deepseek")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an AI assistant. Please answer the question based on the provided context, 
            ensuring that you provide all details. 
            If the answer is not in the context, say "答案不在上下文中" (answer not in context) 
            and do not provide incorrect information."""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [tool]   # Agent requires list[Tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": ques})
    st.write("Answer:", response.get("output", ""))

def user_input(user_question: str):
    if not check_database_exists():
        st.error("Please upload a PDF file and click 'Submit & Process'")
        st.info("Steps: 1. Upload PDF → 2. Process → 3. Ask questions")
        return
    try:
        # Load vector database and create retriever
        db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()

        # Wrap retriever as a tool for the Agent
        retrieval_tool = create_retriever_tool(
            retriever,
            "pdf_extractor",
            "This tool answers queries from the uploaded PDF content."
        )

        # Start the agent and answer
        get_conversational_chain(retrieval_tool, user_question)

    except Exception as e:
        st.error(f"Error while loading database: {e}")
        st.info("Please reprocess the PDF file")

# -------- Streamlit UI --------
def main():
    st.set_page_config(page_title="🤖 RAG chatbot with pdf parser")
    st.header("🤖 RAG chatbot with pdf parser")

    col1, col2 = st.columns([3, 1])
    with col1:
        if not check_database_exists():
            st.warning("⚠️ Please upload and process PDF files first")
    with col2:
        if st.button("🗑️ Clear Database"):
            try:
                import shutil
                if os.path.exists("faiss_db"):
                    shutil.rmtree("faiss_db")
                st.success("Database cleared")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear: {e}")

    # Question input
    user_question = st.text_input(
        "💬 Enter your question",
        placeholder="e.g., What is the main content of this document?",
        disabled=not check_database_exists()
    )
    if user_question:
        if check_database_exists():
            with st.spinner("🤔 AI is analyzing the document..."):
                user_input(user_question)
        else:
            st.error("❌ Please upload and process PDF files first!")

    # Sidebar: Upload & Process
    with st.sidebar:
        st.title("📁 Document Management")
        if check_database_exists():
            st.success("✅ Database Status: Ready")
        else:
            st.info("📝 Status: Waiting for PDF upload")
        st.markdown("---")

        pdf_doc = st.file_uploader("📎 Upload PDF files", accept_multiple_files=True, type=["pdf"], help="Supports multiple PDF files")
        if pdf_doc:
            st.info(f"📄 {len(pdf_doc)} file(s) selected")
            for i, pdf in enumerate(pdf_doc, 1):
                st.write(f"{i}. {pdf.name}")

        if st.button("🚀 Submit & Process", disabled=not pdf_doc, use_container_width=True):
            if pdf_doc:
                with st.spinner("📊 Processing PDF files..."):
                    try:
                        raw_text = pdf_read(pdf_doc)
                        if not raw_text.strip():
                            st.error("❌ Unable to extract text from PDF. Please check if the file is valid")
                            return
                        text_chunks = get_chunks(raw_text)
                        st.info(f"📝 Text split into {len(text_chunks)} chunks")
                        vector_store(text_chunks)
                        st.success("✅ PDF processing complete! You can now start asking questions")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error while processing PDF: {e}")
            else:
                st.warning("⚠️ Please select a PDF file first")

        with st.expander("💡 Instructions"):
            st.markdown(
                """
                **Steps:**
                1. 📎 Upload one or more PDF files  
                2. 🚀 Click "Submit & Process"  
                3. 💬 Enter your question in the main page  
                4. 🤖 AI will answer based on the PDF content  

                **Tips:**
                - Supports multiple PDF files  
                - Large files may take longer to process  
                - You can clear the database anytime to restart  
                """
            )

if __name__ == "__main__":
    main()

