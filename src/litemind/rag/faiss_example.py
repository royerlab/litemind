"""
Example of using the FAISS vector database with litemind RAG.
"""

def faiss_vector_db_example():
    """Example of using the FAISS vector database with litemind RAG."""
    import os
    import tempfile
    
    from litemind import CombinedApi
    from litemind.rag import (
        Document,
        FAISSVectorDatabase,
        RAGAgent,
        create_documents_from_texts,
    )
    
    # Create a temporary directory for the FAISS index
    temp_dir = tempfile.mkdtemp()
    
    # Initialize API
    api = CombinedApi()
    
    # Create a FAISS vector database
    vector_db = FAISSVectorDatabase(
        name="DocumentStore",
        index_folder=temp_dir,
        lazy_loading=True,  # Enable lazy loading
    )
    
    # Create some sample documents
    documents = [
        Document(content="Neural networks are a class of machine learning models inspired by the human brain.", 
                 metadata={"topic": "ai", "subtopic": "neural_networks"}),
        Document(content="Deep learning is a subset of machine learning that uses multi-layered neural networks.", 
                 metadata={"topic": "ai", "subtopic": "deep_learning"}),
        Document(content="Reinforcement learning is learning what to do to maximize a reward signal.", 
                 metadata={"topic": "ai", "subtopic": "reinforcement_learning"}),
        Document(content="Supervised learning involves learning from labeled training data.", 
                 metadata={"topic": "ai", "subtopic": "supervised_learning"}),
        Document(content="Unsupervised learning involves finding patterns in unlabeled data.", 
                 metadata={"topic": "ai", "subtopic": "unsupervised_learning"}),
        Document(content="Natural language processing (NLP) is a field focused on the interaction between computers and human language.", 
                 metadata={"topic": "ai", "subtopic": "nlp"}),
        Document(content="Computer vision is a field that enables computers to derive meaningful information from digital images and videos.", 
                 metadata={"topic": "ai", "subtopic": "computer_vision"}),
    ]
    
    # Add documents to the vector database
    vector_db.add_documents(documents)
    
    # Create a RAG agent
    agent = RAGAgent(
        api=api,
        augmentations=[vector_db],
        k=3,
    )
    
    # Add a system message
    agent.append_system_message("You are a helpful AI assistant who can answer questions about artificial intelligence concepts.")
    
    # Query the agent
    response = agent("What is deep learning and how does it relate to neural networks?")
    
    # Print the response
    print(response[0].to_plain_text())
    
    # Show how the database persists after reloading
    print("\nTesting database persistence by creating a new database instance pointing to the same folder...\n")
    
    # Create a new database instance pointing to the same folder
    new_vector_db = FAISSVectorDatabase(
        name="DocumentStore",  # Same name
        index_folder=temp_dir,  # Same folder
        lazy_loading=True,
    )
    
    # Create a new agent
    new_agent = RAGAgent(
        api=api,
        augmentations=[new_vector_db],
        k=3,
    )
    
    # Add a system message
    new_agent.append_system_message("You are a helpful AI assistant who can answer questions about artificial intelligence concepts.")
    
    # Query the new agent
    response = new_agent("Explain the difference between supervised and unsupervised learning.")
    
    # Print the response
    print(response[0].to_plain_text())
    
    # Clean up the temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    return agent


if __name__ == "__main__":
    faiss_vector_db_example()
