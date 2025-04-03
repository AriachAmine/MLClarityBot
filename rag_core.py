import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_documents(knowledge_base_dir='./knowledge_base'):
    """
    Load all .txt files from the specified knowledge base directory.
    
    Args:
        knowledge_base_dir (str): Path to the directory containing knowledge base text files
        
    Returns:
        list: A list of loaded documents
    """
    try:
        # Check if directory exists
        if not os.path.exists(knowledge_base_dir):
            print(f"Warning: Directory '{knowledge_base_dir}' does not exist.")
            return []
        
        # Use DirectoryLoader with TextLoader for .txt files
        loader = DirectoryLoader(
            knowledge_base_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        # Load documents
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} documents from '{knowledge_base_dir}'")
        return documents
    
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into chunks for better processing.
    
    Args:
        documents (list): List of documents to split
        chunk_size (int): Size of each chunk (default: 500)
        chunk_overlap (int): Overlap between chunks (default: 50)
        
    Returns:
        list: List of document chunks
    """
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def init_embeddings():
    """
    Initialize embeddings using Google Generative AI.
    
    Returns:
        GoogleGenerativeAIEmbeddings: Initialized embeddings object
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not found in environment variables.")
            return None
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        print("Successfully initialized Google Generative AI embeddings")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return None

def create_or_load_vector_store(docs_path='./knowledge_base', index_path='./faiss_index', embeddings=None, force_rebuild=False):
    """
    Create or load a FAISS vector store from documents.
    
    Args:
        docs_path (str): Path to the documents directory
        index_path (str): Path to save/load the FAISS index
        embeddings: The embeddings model to use
        force_rebuild (bool): If True, rebuild the index even if it exists
        
    Returns:
        FAISS: The FAISS vector store
    """
    # If embeddings not provided, initialize them
    if embeddings is None:
        embeddings = init_embeddings()
        if embeddings is None:
            print("Failed to initialize embeddings. Cannot create vector store.")
            return None
    
    # Check if FAISS index already exists and we're not forcing a rebuild
    if os.path.exists(index_path) and not force_rebuild:
        try:
            print(f"Loading existing FAISS index from: {index_path}")
            vector_store = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"Successfully loaded FAISS index with {vector_store.index.ntotal} vectors.")
            return vector_store
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Will attempt to create a new index.")
    else:
        if force_rebuild:
            print("Forcing rebuild of FAISS index...")
    
    # Load and process documents
    docs = load_documents(docs_path)
    if not docs:
        print("No documents to index. Cannot create vector store.")
        return None
    
    # Split documents into chunks
    split_docs = split_documents(docs)
    if not split_docs:
        print("Failed to split documents. Cannot create vector store.")
        return None
    
    # Create FAISS vector store
    try:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # Save the index
        print(f"Saving FAISS index to: {index_path}")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        vector_store.save_local(index_path)
        print(f"Successfully created and saved FAISS index with {vector_store.index.ntotal} vectors.")
        
        return vector_store
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None

def retrieve_context(query, vector_store, k=4):
    """
    Retrieve relevant document chunks from the vector store based on the query.
    
    Args:
        query (str): The user's query text
        vector_store: The FAISS vector store to search in
        k (int): Number of relevant chunks to retrieve (default: 4)
        
    Returns:
        list: List of relevant document contents
    """
    if not vector_store:
        print("Error: No vector store provided for retrieval.")
        return []
    
    try:
        # Perform similarity search
        relevant_docs = vector_store.similarity_search(query, k=k)
        
        # Extract content from documents
        context = [doc.page_content for doc in relevant_docs]
        
        print(f"Retrieved {len(context)} relevant document chunks for query: '{query}'")
        return context
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    docs = load_documents()
    embeddings = init_embeddings()
    if docs and embeddings:
        chunks = split_documents(docs)
        try:
            print(f"First chunk sample: {chunks[0].page_content[:150]}..." if chunks else "No chunks created")
        except Exception as e:
            print(f"Error displaying first chunk: {e}")
    
    embeddings = init_embeddings()
    if embeddings:
        vector_store = create_or_load_vector_store(
            docs_path='./knowledge_base',
            index_path='./faiss_index',
            embeddings=embeddings
        )
        if vector_store:
            print(f"Vector store contains {vector_store.index.ntotal} vectors.")
    
    # Example of context retrieval
    if 'vector_store' in locals() and vector_store:
        retrieved_context = retrieve_context(
            query="What is overfitting and how can I prevent it?",
            vector_store=vector_store
        )
        
        # Print a sample of the retrieved context
        if retrieved_context:
            print("\nSample of retrieved context:")
            print(f"{retrieved_context[0][:150]}...\n")
