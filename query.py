from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from dotenv import load_dotenv
from openai import OpenAI
import sys

load_dotenv()

# Connect to the local llama.cpp server
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="none" 
)

MODEL = "local-model" 

def main():
    print("Loading local embeddings and models...")
    
    # 1. Load your standard File Clerk (Embeddings)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    db = Chroma(
        persist_directory="clinical_db",
        embedding_function=embeddings
    )

    # 2. STAGE 1 RETRIEVAL: Fetch a broad net of 10 chunks
    base_retriever = db.as_retriever(search_kwargs={"k": 10})

    # 3. STAGE 2 RETRIEVAL: Load the Local Re-Ranker model
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print("\nAdvanced Clinical RAG (with Custom Re-Ranker) Initialized. Type 'exit' to quit.\n")

    while True:
        question = input("Clinical query: ")
        if question.lower() in ['exit', 'quit']:
            break
        if not question.strip():
            continue

        # --- THE CUSTOM RE-RANKING ALGORITHM ---
        
        # Step A: Fetch top 10 chunks based on fast vector search
        initial_docs = base_retriever.invoke(question)
        
        # Step B: Create [Question, Chunk] pairs for the Cross-Encoder to read
        scoring_pairs = [[question, doc.page_content] for doc in initial_docs]
        
        # Step C: The Cross-Encoder scores each pair based on exact semantic relevance
        scores = cross_encoder.score(scoring_pairs)
        
        # Step D: Sort the documents by their new scores (highest to lowest)
        scored_docs = list(zip(initial_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Step E: Keep only the absolute best 3 documents
        best_docs = [doc for doc, score in scored_docs[:3]]
        
        # ---------------------------------------

        formatted_context = ""
        for i, doc in enumerate(best_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            page_num = doc.metadata.get('page', 'Unknown Page')
            if isinstance(page_num, int):
                page_num += 1 

            formatted_context += f"--- START CHUNK {i+1} ---\n"
            formatted_context += f"[CITATION: {source}, Page {page_num}]\n"
            formatted_context += f"{doc.page_content}\n"
            formatted_context += f"--- END CHUNK {i+1} ---\n\n"

        combined_prompt = f"""Clinical context:
{formatted_context}

Task: You are a clinical assistant. Answer the question using ONLY the context above. 
Rule 1: If the answer is not in the context, say "Insufficient data in the clinical repository."
Rule 2: Be concise and use bullet points.
Rule 3:  Explain your reasoning
Rule 4: Cite your claims using the exact [CITATION: ...] tags provided at the end of sentences.

Question: {question}

Answer:"""
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.1,  
                max_tokens=1024, 
                stream=True      
            )
            
            print("\n", end="")
            
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
            
            print("\n")
            
        except Exception as e:
            print(f"\n[!] Error connecting to local server: {e}\n")
            
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()