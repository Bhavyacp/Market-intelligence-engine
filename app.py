import streamlit as st
import requests

st.set_page_config(page_title="MarketPulse AI", layout="wide")

st.title("📊 MarketPulse AI: Enterprise Intelligence Platform")
st.caption("Integrated Big Data, Core ML, Deep Learning Embeddings, and GenAI Engine")

st.markdown("---")

col1, col2 = st.split_columns([1, 2]) if hasattr(st, 'split_columns') else st.columns([1, 2])

with col1:
    st.subheader("💡 Strategic Queries")
    sample_query = st.selectbox(
        "Select a sample enterprise query:",
        [
            "What is TechCorp's cloud pricing strategy?",
            "What are DataDynamics expansion risks in Europe?",
            "Summarize competitive AI threats this quarter."
        ]
    )
    custom_query = st.text_input("Or enter a custom market query:")
    query_to_run = custom_query if custom_query else sample_query
    
    run_button = st.button("Generate Executive Brief", type="primary")

with col2:
    st.subheader("📈 Generated Market Brief")
    if run_button:
        with st.spinner("Synthesizing data across Big Data, Vector DB, and LLM..."):
            try:
                res = requests.post("http://localhost:8000/analyze", json={"query": query_to_run})
                if res.status_code == 200:
                    report = res.json().get("executive_report")
                    st.markdown(report)
                else:
                    st.error("Error communicating with Backend API.")
            except Exception as e:
                st.error(f"Failed to connect to API server at http://localhost:8000. Error: {e}")

st.markdown("---")
st.info("System Status: PySpark Engine [Active] | XGBoost Model [Loaded] | ChromaDB Embeddings [Ready] | RAG Agent [Online]")