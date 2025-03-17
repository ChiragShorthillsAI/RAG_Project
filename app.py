import streamlit as st
import logging
import os
from dotenv import load_dotenv

# Set Streamlit page config as the first command
st.set_page_config(page_title="Q&A Interface", layout="centered")

# Pinecone + LangChain Imports
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

# Load environment variables
load_dotenv()

# Suppress library debug logs
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

# Set up a dedicated QA logger
qa_logger = logging.getLogger("qa_logger")
qa_logger.setLevel(logging.INFO)
if not qa_logger.handlers:
    qa_file_handler = logging.FileHandler("/home/shtlp_0132/Desktop/Work/RAG_Project/qa_interactions.log", mode='a')
    qa_file_formatter = logging.Formatter("%(asctime)s - QUESTION: %(message)s | ANSWER: %(message)s")
    qa_file_handler.setFormatter(qa_file_formatter)
    qa_logger.addHandler(qa_file_handler)

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Connect to Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY not found. Make sure it's set in your .env file.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag"
dimension = 768  # Google Generative AI embeddings

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

@st.cache_resource
def load_docsearch():
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embed_model)

docsearch = load_docsearch()

@st.cache_resource
def create_rag_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Provide as much detail as needed.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = create_rag_chain()

@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(ResourceExhausted))
def safe_invoke(question: str):
    return rag_chain.invoke({"input": question})

def get_answer(question: str) -> str:
    try:
        response = safe_invoke(question)
        answer = response.get("answer", "No answer available.").replace("\n", " ").replace("\r", " ")
        qa_logger.info(f"{question} | ANSWER: {answer}")
        return answer
    except Exception as e:
        return f"Error retrieving answer: {str(e)}"

# Streamlit UI - Dark Mode Chat Interface
st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
        .stApp { background-color: #121212; color: white; }
        .chat-container { max-width: 700px; margin: auto; }
        .chat-history { max-height: 500px; overflow-y: auto; padding: 15px; border-radius: 10px; }
        .chat-bubble { border-radius: 20px; padding: 10px 15px; max-width: 70%; color: white; }
        .user-bubble { background-color: #1e88e5; align-self: flex-end; margin-left: auto; }
        .bot-bubble { background-color: #424242; align-self: flex-start; }
        .avatar { width: 40px; height: 40px; border-radius: 50%; margin-left: 10px; }
        .chat-input { position: sticky; bottom: 10px; background: #212121; color: white; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("Q&A Interface")
st.write("Ask a question and get an AI-powered answer.")

# Chat history container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("<div class='chat-history'>", unsafe_allow_html=True)

for i, (q, a) in enumerate(st.session_state.chat_history):
    user_avatar = "👤"
    bot_avatar = "🤖"
    
    # User Question (Align Right)
    st.markdown(f"<div style='display: flex; justify-content: flex-end; align-items: center;'>"
                f"<div class='chat-bubble user-bubble'>{q}</div>"
                f"<span class='avatar'>{user_avatar}</span>"
                f"</div>", unsafe_allow_html=True)
    
    # Bot Answer (Align Left)
    st.markdown(f"<div style='display: flex; align-items: center;'>"
                f"<span class='avatar'>{bot_avatar}</span>"
                f"<div class='chat-bubble bot-bubble'>{a}</div>"
                f"</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Chat input field at the bottom
question = st.text_input("Message:", key="question_input")
if st.button("Send") or (question and st.session_state.get("enter_pressed")):
    if question:
        with st.spinner("Thinking..."):
            answer = get_answer(question)
        st.session_state.chat_history.append((question, answer))
        st.rerun()

st.markdown("<p style='text-align:center; font-size: 12px; color: white;'>Made with Streamlit</p>", unsafe_allow_html=True)