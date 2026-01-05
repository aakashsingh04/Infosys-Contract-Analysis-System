"""Streamlit UI for Contract Analysis."""
import streamlit as st
import tempfile
import os
from pathlib import Path
from contract_analyzer import ContractAnalyzer
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

st.set_page_config(page_title="Contract Analysis", page_icon="📄", layout="wide")

st.title("📄 Contract Analysis System")
st.markdown("Upload a contract to analyze with 4 AI agents: Compliance, Finance, Legal, Operations")

@st.cache_resource
def get_analyzer():
    return ContractAnalyzer(use_free_model=True)

if 'document_history' not in st.session_state:
    st.session_state.document_history = []

if 'stored_results' not in st.session_state:
    st.session_state.stored_results = {}

analyzer = get_analyzer()

# Sidebar with history
with st.sidebar:
    st.header("📚 Document History")
    
    if st.session_state.document_history:
        for idx, doc in enumerate(st.session_state.document_history):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{doc['name'][:40]}**")
                st.caption(f"📅 {doc['date']}")
            with col2:
                if st.button("View", key=f"view_{idx}"):
                    st.session_state.selected_doc_id = doc['id']
                    st.rerun()
            st.divider()
    else:
        st.info("No documents uploaded yet")

# Main content area
uploaded_file = st.file_uploader("Upload Contract (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Semantic search
with st.expander("🔎 Search across uploaded documents"):
    query = st.text_input("Enter search query")
    top_k = st.slider("Results", 1, 10, 5)
    if st.button("Search"):
        if query.strip():
            try:
                results = analyzer.search(query, k=top_k)
                if not results:
                    st.info("No matches found.")
                else:
                    for idx, r in enumerate(results, 1):
                        st.markdown(f"**{idx}. Doc:** {r.get('document_id','')}")
                        st.write(r.get("text","")[:400] + "...")
                        st.divider()
            except Exception as e:
                st.error(f"Search error: {e}")
        else:
            st.warning("Enter a query first.")

# Display selected history document
if 'selected_doc_id' in st.session_state:
    doc_id = st.session_state.selected_doc_id
    if doc_id in st.session_state.stored_results:
        results = st.session_state.stored_results[doc_id]
        doc_info = next((d for d in st.session_state.document_history if d['id'] == doc_id), None)
        
        if doc_info:
            st.header(f"📄 {doc_info['name']}")
            st.caption(f"Uploaded: {doc_info['date']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔒 Compliance Analysis")
                st.write(results['analyses']['compliance']['analysis'])
                
                st.subheader("💰 Finance Analysis")
                st.write(results['analyses']['finance']['analysis'])
            
            with col2:
                st.subheader("⚖️ Legal Analysis")
                st.write(results['analyses']['legal']['analysis'])
                
                st.subheader("⚙️ Operations Analysis")
                st.write(results['analyses']['operations']['analysis'])
            
            if st.button("Back to Upload"):
                del st.session_state.selected_doc_id
                st.rerun()

# Upload and analyze new document
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    if st.button("Analyze Contract", type="primary"):
        with st.spinner("Analyzing with AI agents..."):
            try:
                document_id = analyzer.upload_document(tmp_path)
                results = analyzer.analyze_contract(document_id)
                
                # Store in history
                st.session_state.document_history.insert(0, {
                    'id': document_id,
                    'name': uploaded_file.name,
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'metadata': results.get('document_metadata', {})
                })
                
                # Store results
                st.session_state.stored_results[document_id] = results
                
                # Keep only last 20
                if len(st.session_state.document_history) > 20:
                    old_doc = st.session_state.document_history.pop()
                    if old_doc['id'] in st.session_state.stored_results:
                        del st.session_state.stored_results[old_doc['id']]
                
                st.success("✅ Analysis Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔒 Compliance Analysis")
                    st.write(results['analyses']['compliance']['analysis'])
                    
                    st.subheader("💰 Finance Analysis")
                    st.write(results['analyses']['finance']['analysis'])
                
                with col2:
                    st.subheader("⚖️ Legal Analysis")
                    st.write(results['analyses']['legal']['analysis'])
                    
                    st.subheader("⚙️ Operations Analysis")
                    st.write(results['analyses']['operations']['analysis'])
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("💡 Get free Groq API key: https://console.groq.com/")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
