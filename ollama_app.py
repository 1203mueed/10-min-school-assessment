import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import ollama
import os
from create_vector_db import get_vectorstore, DB_DIR, EMBEDDING_MODEL

# Initialize models
LLM_MODEL = "gemma3:12b"

# Set up Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot (Streaming)")

# Custom CSS for dynamic background colors
st.markdown("""
    <style>
    /* Style for assistant's response while streaming (typing) */
    .streaming {
        background-color: #FFFACD; /* Light yellow for typing */
        padding: 10px;
        border-radius: 5px;
    }
    /* Style for assistant's response when complete */
    .complete {
        background-color: #E0FFE0; /* Light green for complete */
        padding: 10px;
        border-radius: 5px;
    }
    /* Style for thinking indicator */
    .think {
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize vector store in session state to avoid reloading
if "vectorstore" not in st.session_state:
    with st.spinner("🔄 Loading vector database..."):
        vectorstore = get_vectorstore()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.success("✅ Vector database loaded successfully!")
        else:
            st.error("❌ Failed to load vector database!")
            st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your RAG chatbot. Ask me questions regarding documents."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f'<div class="complete">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Function to retrieve context
def retrieve_context(query: str) -> str:
    results = st.session_state.vectorstore.similarity_search(query, k=5)
    return "\n\n".join([doc.page_content for doc in results])

# Function to generate streaming answer
def generate_answer(query: str, context: str):
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": f"প্রসঙ্গটি পড়ো এবং প্রশ্নের সংক্ষিপ্ত ও সরাসরি উত্তর দাও।\n\nপ্রসঙ্গ:\n{context}"},
            {"role": "user", "content": query}
        ],
        stream=True
    )
    return stream

# Input field for user messages
if user_input := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate streaming response
    try:
        context = retrieve_context(user_input)
        
        # Create a placeholder for the streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            stream = generate_answer(user_input, context)
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(
                        f'<div class="streaming">{full_response}▌</div>',
                        unsafe_allow_html=True
                    )
            
            # Finalize the response with complete style
            response_placeholder.markdown(
                f'<div class="complete">{full_response}</div>',
                unsafe_allow_html=True
            )
        
        # Add the complete assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
            
    except Exception as e:
        st.error(f"Error communicating with {LLM_MODEL}: {str(e)}")
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again!"})
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="complete">Sorry, I encountered an error. Please try again!</div>',
                unsafe_allow_html=True
            )