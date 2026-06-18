from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Updated import!
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Using the absolute path you confirmed earlier
PDF_PATH = "C:/Users/Venura/Desktop/VEditor/Data/GINA-2025-Update-25_11_08-WMS.pdf"

def main():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    print("Generating local embeddings (this may take a moment to download the model on the first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Saving to Chroma vector database...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="clinical_db"
    )

    print("RAG ingestion complete. Ready for local querying!")

if __name__ == "__main__":
    main()