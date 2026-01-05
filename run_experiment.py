import os
import json
from contract_analyzer import ContractAnalyzer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_experiment():
    print("Starting contract analysis experiment...")
    
    # Initialize analyzer
    try:
        analyzer = ContractAnalyzer()
        print(f"Analyzer initialized with vector store: {analyzer.vector_store_type}")
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return

    # Sample contract path
    sample_path = os.path.abspath("experiments/sample_contract.txt")
    if not os.path.exists(sample_path):
        print(f"Sample contract not found at {sample_path}")
        return

    # Upload document
    print(f"\nUploading document: {sample_path}")
    try:
        doc_id = analyzer.upload_document(sample_path)
        print(f"Document uploaded successfully. ID: {doc_id}")
    except Exception as e:
        print(f"Failed to upload document: {e}")
        return

    # Analyze document
    print("\nRunning analysis (this may take a minute)...")
    try:
        # Define roles to test
        roles = ["compliance", "legal", "finance", "operations"]
        result = analyzer.analyze_contract(doc_id, agent_roles=roles)
        
        print("\nAnalysis Complete!")
        print("-" * 50)
        
        # Print summary
        if "summary" in result:
            print("\nExecutive Summary:")
            print(result["summary"])
            
        # Print risks
        if "risks" in result:
            print("\nIdentified Risks:")
            for risk in result["risks"]:
                print(f"- [{risk.get('severity', 'Unknown')}] {risk.get('description', 'No description')}")
                
        # Save full result to file
        output_path = "experiments/analysis_result.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull results saved to {output_path}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    run_experiment()
