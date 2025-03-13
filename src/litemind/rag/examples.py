"""
Examples of using the litemind RAG functionality.
"""

def basic_rag_example():
    """A basic example of using RAG with litemind."""
    from litemind import CombinedApi
    from litemind.rag import (
        Document,
        InMemoryVectorDatabase,
        RAGAgent,
        create_documents_from_texts,
    )
    
    # Initialize API
    api = CombinedApi()
    
    # Create a simple in-memory vector database
    vector_db = InMemoryVectorDatabase(name="Knowledge Base")
    
    # Add some documents
    texts = [
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
        "Einstein is best known for his theory of relativity, including the famous equation E=mc².",
        "In 1921, Albert Einstein received the Nobel Prize in Physics for his discovery of the law of the photoelectric effect.",
        "Einstein moved to the United States in 1933 and became a professor at Princeton University.",
        "Albert Einstein died on April 18, 1955, in Princeton, New Jersey, at the age of 76.",
    ]
    
    documents = create_documents_from_texts(texts)
    vector_db.add_documents(documents)
    
    # Create a RAG agent
    agent = RAGAgent(
        api=api,
        augmentations=[vector_db],
        k=3,  # Number of documents to retrieve
        context_position="before_query",  # Add context before the query
    )
    
    # Add a system message
    agent.append_system_message("You are a helpful assistant who answers questions about Albert Einstein.")
    
    # Query the agent
    response = agent("When was Einstein born?")
    
    # Print the response
    print(response[0].to_plain_text())
    
    return agent


def hierarchical_document_example():
    """Example of using hierarchical documents with litemind RAG."""
    import os
    
    from litemind import CombinedApi
    from litemind.rag import (
        DocumentLoader,
        InMemoryVectorDatabase,
        RAGAgent,
    )
    
    # Initialize API
    api = CombinedApi()
    
    # Create a vector database
    vector_db = InMemoryVectorDatabase(name="Document Database")
    
    # Load a document with hierarchical structure
    sample_text_path = os.path.join(os.path.dirname(__file__), "sample.txt")
    
    # Create the sample text file if it doesn't exist
    if not os.path.exists(sample_text_path):
        sample_text = """# Introduction to Machine Learning

Machine learning is a subfield of artificial intelligence that focuses on developing systems that learn from data.

## Supervised Learning

Supervised learning involves training a model on a labeled dataset, where each example has an input and a corresponding output.

### Classification

Classification is a supervised learning task where the output is a categorical variable.

### Regression

Regression is a supervised learning task where the output is a continuous variable.

## Unsupervised Learning

Unsupervised learning involves training a model on an unlabeled dataset, where the examples have inputs but no corresponding outputs.

### Clustering

Clustering is an unsupervised learning task that involves grouping similar examples.

### Dimensionality Reduction

Dimensionality reduction is an unsupervised learning task that involves reducing the number of input variables.
"""
        with open(sample_text_path, "w") as f:
            f.write(sample_text)
    
    # Load the document
    document = DocumentLoader.load_from_file(
        sample_text_path,
        split_documents=True,
        chunk_size=200,
        chunk_overlap=50,
    )
    
    # Get all flat documents from the hierarchy
    flat_docs = document.get_all_documents()
    
    # Add documents to the vector database
    vector_db.add_documents(flat_docs)
    
    # Create a RAG agent
    agent = RAGAgent(
        api=api,
        augmentations=[vector_db],
        k=3,
    )
    
    # Query the agent
    response = agent("What is clustering in machine learning?")
    
    # Print the response
    print(response[0].to_plain_text())
    
    return agent


def react_rag_example():
    """Example of using RAG with ReAct Agent."""
    from datetime import datetime
    
    from litemind import CombinedApi
    from litemind.agent.tools.toolset import ToolSet
    from litemind.rag import (
        Document,
        InMemoryVectorDatabase,
        ReActRAGAgent,
        create_documents_from_texts,
    )
    
    # Initialize API
    api = CombinedApi()
    
    # Create a vector database
    vector_db = InMemoryVectorDatabase(name="Knowledge Base")
    
    # Add some documents
    texts = [
        "The capital of France is Paris.",
        "The capital of Italy is Rome.",
        "The capital of Germany is Berlin.",
        "The capital of Spain is Madrid.",
        "The capital of Portugal is Lisbon.",
    ]
    
    documents = create_documents_from_texts(texts)
    vector_db.add_documents(documents)
    
    # Define some tools
    def get_current_date():
        """Get the current date."""
        return datetime.now().strftime("%Y-%m-%d")
    
    def calculate(expression: str):
        """Calculate the result of a mathematical expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"
    
    # Create a toolset
    toolset = ToolSet()
    toolset.add_function_tool(get_current_date, "Get the current date")
    toolset.add_function_tool(calculate, "Calculate the result of a mathematical expression")
    
    # Create a ReActRAGAgent
    agent = ReActRAGAgent(
        api=api,
        augmentations=[vector_db],
        toolset=toolset,
        k=3,
        max_reasoning_steps=5,
    )
    
    # Add a system message
    agent.append_system_message("You are a helpful assistant who answers questions about European capitals and can also perform calculations and return the current date.")
    
    # Query the agent
    response = agent("What's the capital of France and what's today's date?")
    
    # Print the response
    print(response.to_plain_text())
    
    return agent


def multiple_augmentations_example():
    """Example of using multiple augmentations with litemind RAG."""
    from litemind import CombinedApi
    from litemind.rag import (
        AugmentationSet,
        Document,
        InMemoryVectorDatabase,
        RAGAgent,
        create_documents_from_texts,
    )
    
    # Initialize API
    api = CombinedApi()
    
    # Create two vector databases for different knowledge domains
    science_db = InMemoryVectorDatabase(name="Science Knowledge")
    history_db = InMemoryVectorDatabase(name="History Knowledge")
    
    # Add science documents
    science_texts = [
        "The theory of relativity was developed by Albert Einstein.",
        "Quantum mechanics is a fundamental theory in physics that describes nature at the atomic and subatomic scales.",
        "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix.",
        "The periodic table is a tabular display of the chemical elements, organized based on their atomic numbers.",
    ]
    
    science_docs = create_documents_from_texts(science_texts)
    science_db.add_documents(science_docs)
    
    # Add history documents
    history_texts = [
        "World War I began in 1914 and ended in 1918.",
        "The French Revolution started in 1789 with the storming of the Bastille.",
        "The Roman Empire was founded in 27 BCE by Augustus Caesar.",
        "The United States Declaration of Independence was adopted on July 4, 1776.",
    ]
    
    history_docs = create_documents_from_texts(history_texts)
    history_db.add_documents(history_docs)
    
    # Create an augmentation set
    augmentation_set = AugmentationSet([science_db, history_db])
    
    # Create a RAG agent
    agent = RAGAgent(
        api=api,
        augmentations=augmentation_set.augmentations,
        k=2,  # Retrieve 2 documents from each augmentation
    )
    
    # Add a system message
    agent.append_system_message("You are a helpful assistant who can answer questions about science and history.")
    
    # Query the agent
    response = agent("Tell me about DNA and the Roman Empire.")
    
    # Print the response
    print(response[0].to_plain_text())
    
    return agent


if __name__ == "__main__":
    # Run one of the examples
    # basic_rag_example()
    # hierarchical_document_example()
    react_rag_example()
    # multiple_augmentations_example()
