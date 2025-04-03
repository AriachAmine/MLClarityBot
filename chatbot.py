import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

def init_gemini_chat_model(model_name="gemini-1.5-flash-latest", temperature=0.3):
    """
    Initialize and return a ChatGoogleGenerativeAI model.
    
    Args:
        model_name (str): Name of the Gemini model to use
        temperature (float): Temperature setting for generation (lower = more factual)
        
    Returns:
        ChatGoogleGenerativeAI: Initialized chat model, or None if initialization fails
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        return None
    
    try:
        chat_model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        print(f"Successfully initialized Gemini chat model: {model_name}")
        return chat_model
    except Exception as e:
        print(f"Error initializing Gemini chat model: {e}")
        return None

def create_rag_prompt_template():
    """
    Create a ChatPromptTemplate for the RAG system.
    
    Returns:
        ChatPromptTemplate: The configured prompt template
    """
    system_template = """You are ML ClarityBot, a specialized AI assistant that explains machine learning concepts clearly and concisely.

INSTRUCTIONS:
1. Answer ONLY based on the provided context below. Do not use your general knowledge.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question properly. Please ask about one of the machine learning concepts in our knowledge base."
3. Provide clear, concise explanations using simple language.
4. Always include a simple, relatable real-world analogy to illustrate the concept.
5. Structure your response clearly with appropriate headings when helpful.
6. DO NOT mention that you're using "context" or "retrieved information" in your answer.
7. VERY IMPORTANT - If the context mentions or implies a diagram would be helpful, create a simple Mermaid diagram code within a markdown code block (```mermaid ... ```). Keep diagrams simple (flowchart TD, comparison, etc.) Only include a diagram if it adds real value.

MERMAID DIAGRAM RULES:
- Always start with a valid diagram type: flowchart TD, flowchart LR, classDiagram, etc.
- Use simple syntax and keep diagrams concise
- Ensure all nodes are properly defined before they're referenced
- For flowcharts, use this structure:
  ```mermaid
  flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
  ```
- Test your diagram mentally to ensure it's valid Mermaid syntax

CONTEXT:
{context}
"""

    human_template = "Please explain the following machine learning concept: {question}"
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt

def create_rag_chain(vector_store, llm, prompt_template):
    """
    Create a RAG chain using LangChain Expression Language (LCEL).
    
    Args:
        vector_store: FAISS vector store with the knowledge base
        llm: Initialized LLM (Gemini model)
        prompt_template: ChatPromptTemplate for structuring the prompt
        
    Returns:
        Runnable: The RAG chain that can be invoked with a question
    """
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Define how to retrieve context
    def retrieve_context(question):
        docs = retriever.invoke(question)
        # Join the content of all retrieved documents
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build the RAG chain using LCEL
    rag_chain = (
        {"context": lambda question: retrieve_context(question),
         "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    # Test model initialization
    chat_model = init_gemini_chat_model()
    
    if chat_model:
        print("Testing chat model with a simple query...")
        try:
            messages = [
                SystemMessage(content="You are a helpful AI assistant focused on explaining machine learning concepts clearly."),
                HumanMessage(content="What is the difference between supervised and unsupervised learning?")
            ]
            response = chat_model.invoke(messages)
            print(f"Response: {response.content[:150]}...")  # Print first 150 chars
        except Exception as e:
            print(f"Error during model invocation: {e}")
        
        print("Testing prompt template creation...")
        prompt_template = create_rag_prompt_template()
        print("Successfully created RAG prompt template")
        
        # Example test with dummy context and question
        formatted_prompt = prompt_template.format_messages(
            context="Overfitting occurs when a model learns the training data too well, capturing noise rather than the underlying pattern. It performs well on training data but poorly on new data. Cross-validation helps detect overfitting by testing on held-out data.",
            question="What is overfitting and how can I prevent it?"
        )
        print(f"Number of messages in formatted prompt: {len(formatted_prompt)}")
    
    # Example of creating a RAG chain (only run if vector_store exists)
    from rag_core import create_or_load_vector_store, init_embeddings
    
    print("Testing RAG chain creation...")
    try:
        # Initialize components
        chat_model = init_gemini_chat_model()
        prompt_template = create_rag_prompt_template()
        embeddings = init_embeddings()
        
        if chat_model and prompt_template and embeddings:
            # Get or create vector store
            vector_store = create_or_load_vector_store(embeddings=embeddings)
            
            if vector_store:
                # Create the RAG chain
                rag_chain = create_rag_chain(
                    vector_store=vector_store,
                    llm=chat_model,
                    prompt_template=prompt_template
                )
                
                print("Successfully created RAG chain. Testing with a sample question...")
                
                # Test the chain
                sample_question = "What is overfitting and how can I prevent it?"
                response = rag_chain.invoke(sample_question)
                
                print("\nSample response:")
                print(f"{response[:300]}...")  # Print first 300 chars of the response
            else:
                print("Failed to create or load vector store.")
        else:
            print("Failed to initialize one or more required components.")
    except Exception as e:
        print(f"Error creating or testing RAG chain: {e}")
    
    print("\n" + "="*50)
    print("COMPLETE RAG CHAIN DEMONSTRATION")
    print("="*50)
    
    try:
        # Initialize the LLM
        llm = init_gemini_chat_model(temperature=0.2)  # Lower temperature for more factual responses
        
        # Create the prompt template
        prompt_template = create_rag_prompt_template()
        
        # Initialize embeddings
        from rag_core import init_embeddings, create_or_load_vector_store
        embeddings = init_embeddings()
        
        if llm and prompt_template and embeddings:
            print("Loading vector store from FAISS index...")
            # Load the vector store (or create if it doesn't exist)
            vector_store = create_or_load_vector_store(
                docs_path='./knowledge_base',
                index_path='./faiss_index',
                embeddings=embeddings
            )
            
            if vector_store:
                print(f"Successfully loaded vector store with {vector_store.index.ntotal} vectors")
                
                # Create the RAG chain
                rag_chain = create_rag_chain(
                    vector_store=vector_store,
                    llm=llm,
                    prompt_template=prompt_template
                )
                
                # Define a sample question that specifically asks for an analogy and diagram
                sample_question = "Explain overfitting using an analogy and a simple diagram if possible"
                print(f"\nAsking: '{sample_question}'")
                print("\nGenerating response (this may take a moment)...")
                
                # Invoke the chain
                response = rag_chain.invoke(sample_question)
                
                # Print the full response
                print("\n" + "="*50)
                print("GENERATED RESPONSE:")
                print("="*50)
                print(response)
                print("="*50)
            else:
                print("Failed to load vector store. Make sure you've generated the knowledge base first.")
        else:
            print("Failed to initialize one or more components required for the RAG chain.")
    except Exception as e:
        print(f"Error in RAG chain demonstration: {e}")
