from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

# Configuration
EMBEDDING_MODEL = "snowflake-arctic-embed2"
DB_DIR = "db_full_story"
TEXT_SOURCE = "data/output.txt"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

def create_vector_database():
    """
    Create or load the vector database from the text source
    """
    print("🔄 Starting vector database creation...")
    
    try:
        # Load the document
        print(f"📕 Loading document from: {TEXT_SOURCE}")
        loader = TextLoader(TEXT_SOURCE, encoding="utf-8")
        docs = loader.load()
        print(f"✅ Document loaded successfully. Pages: {len(docs)}")

        # Split the document into chunks
        print("✂️ Splitting document into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        print(f"✅ Document split into {len(chunks)} chunks")

        # Load embeddings
        print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # Check if vector store already exists
        if os.path.exists(f"{DB_DIR}/chroma.sqlite3"):
            print("🔑 Vector database already exists! Loading it...")
            vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            print("✅ Vector database loaded successfully")
        else:
            print("📦 Creating new vector database...")
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
            print("✅ Vector database created successfully")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating vector database: {str(e)}")
        return None

def get_vectorstore():
    """
    Get the vector store instance (create if doesn't exist)
    """
    return create_vector_database()

if __name__ == "__main__":
    # Run this script directly to create/update the vector database
    print("🚀 Vector Database Creation Script")
    print("=" * 50)
    
    vectorstore = create_vector_database()
    
    if vectorstore:
        print("\n🎉 Vector database is ready!")
        print(f"📁 Database location: {DB_DIR}")
    else:
        print("\n💥 Failed to create vector database!") 