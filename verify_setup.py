import os
from contract_analyzer import ContractAnalyzer
from dotenv import load_dotenv

load_dotenv()

print("Initializing ContractAnalyzer...")
try:
    analyzer = ContractAnalyzer()
    print(f"Vector Store Type: {analyzer.vector_store_type}")
    
    if analyzer.vector_store_type == "pinecone":
        print("SUCCESS: Pinecone vector store initialized.")
    else:
        print(f"WARNING: Vector store is {analyzer.vector_store_type}, expected 'pinecone'.")
        
except Exception as e:
    print(f"ERROR: Initialization failed: {e}")
