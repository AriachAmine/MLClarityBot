import streamlit as st
import re
from streamlit_mermaid import st_mermaid
from chatbot import init_gemini_chat_model, create_rag_prompt_template, create_rag_chain
from rag_core import init_embeddings, create_or_load_vector_store

# Cache resource-intensive operations
@st.cache_resource
def initialize_rag_components():
    """Initialize and cache the RAG components for efficiency"""
    llm = init_gemini_chat_model(temperature=0.2)
    prompt_template = create_rag_prompt_template()
    embeddings = init_embeddings()
    vector_store = create_or_load_vector_store(
        docs_path='./knowledge_base',
        index_path='./faiss_index',
        embeddings=embeddings
    )
    
    if all([llm, prompt_template, embeddings, vector_store]):
        rag_chain = create_rag_chain(vector_store, llm, prompt_template)
        return {
            "rag_chain": rag_chain,
            "initialized": True
        }
    return {"initialized": False}

# Initialize components when the app starts
rag_components = initialize_rag_components()

# Set page title and description
st.title("ML Clarity Bot")
st.markdown("*Ask me about any machine learning concept, and I'll explain it clearly with analogies and diagrams when helpful.*")

# Create a text input for the user's question
user_question = st.text_input("Enter your machine learning question:", placeholder="E.g., What is overfitting?")

# Create a button to submit the question
explain_button = st.button("Explain")

# Create a placeholder for the answer
answer_container = st.container()

# Function to detect and handle Mermaid diagrams in the response
def process_response_with_mermaid(response):
    # Regular expression to find Mermaid code blocks
    mermaid_pattern = r"```mermaid\n(.*?)```"
    
    # Search for Mermaid code blocks using regex
    match = re.search(mermaid_pattern, response, re.DOTALL)
    
    if match:
        # Extract the Mermaid code (without the fences)
        mermaid_code = match.group(1).strip()
        
        # Split the response into parts
        parts = re.split(mermaid_pattern, response, maxsplit=1, flags=re.DOTALL)
        
        # Display text before the Mermaid block
        if parts[0].strip():
            st.markdown(parts[0].strip())
        
        # Display the Mermaid diagram
        st.write("### Diagram:")
        st_mermaid(mermaid_code)
        
        # Display text after the Mermaid block (if any)
        if len(parts) > 2 and parts[2].strip():
            st.markdown(parts[2].strip())
    else:
        # No Mermaid blocks found, display the entire response
        st.markdown(response)

# Process the user's question when the button is clicked
if explain_button:
    if not user_question:
        st.warning("Please enter a machine learning concept to explain.")
    elif not rag_components["initialized"]:
        st.error("Failed to initialize the system. Please check if the knowledge base is generated.")
    else:
        with st.spinner("Generating explanation..."):
            # Get the RAG chain from the cached components
            rag_chain = rag_components["rag_chain"]
            
            # Invoke the chain with the user's question
            response = rag_chain.invoke(user_question)
            
            # Display the response in the container
            with answer_container:
                process_response_with_mermaid(response)
