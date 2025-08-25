import os
import re
import string
from openai import OpenAI 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from datetime import datetime
from langchain_core.documents import Document
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, CSVLoader

load_dotenv()  # Load environment variables from .env


def clean_text(text: str, lowercase: bool = True, remove_punct: bool = False) -> str:
    """
    Cleans extracted text for preprocessing:
    - Lowercase (optional)
    - Remove line breaks, tabs
    - Remove punctuation (optional)
    - Normalize spaces
    """
    if not text:
        return ""
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Replace newlines and tabs with space 
    text = text.replace("\n", " ").replace("\t", " ")
    
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

# -------------------------------
# File loader
# -------------------------------
def load_files(path: str) -> list[Document]:
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        reader = PdfReader(path)
        all_text = "".join((p.extract_text() or "") for p in reader.pages)
        cleaned = clean_text(all_text, lowercase=True, remove_punct=False)
        return [Document(page_content=cleaned, metadata={"source": path})]

    elif file_extension == '.txt':
        docs = TextLoader(path, encoding='utf8').load()
        for d in docs:
            d.page_content = clean_text(d.page_content)
        return docs

    elif file_extension == '.docx':
        docs = Docx2txtLoader(path).load()
        for d in docs:
            d.page_content = clean_text(d.page_content)
        return docs
        
    elif file_extension == '.csv':
        docs = CSVLoader(path).load()
        for d in docs:
            d.page_content = clean_text(d.page_content)
        return docs

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
# Load the collection
def load_chroma_collection(name: str, directory: str) -> Chroma:
    """
    Load an existing Chroma collection.

    Args:
        name (str): Name of the collection.
        directory (str): Directory where the collection is persisted.

    Returns:
        Chroma: The loaded Chroma vectorstore.
    """
    persist_directory = os.path.join(directory, name)
    if not os.path.exists(persist_directory):
        raise ValueError(f"Collection '{name}' does not exist in '{directory}'.")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    collection = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return collection

# Add documents to the collection
def add_documents_to_collection(collection: Chroma, new_documents: List[Document]) -> None:
    """
    Add new documents to an existing Chroma collection.

    Args:
        collection (Chroma): The Chroma vectorstore to add documents to.
        new_documents (List[Document]): List of new LangChain Document objects to add.
    """
    if not new_documents:
        print("No new documents to add.")
        return

    collection.add_documents(new_documents)
    collection.persist()
    print(f"Added {len(new_documents)} documents to the collection and persisted changes.")
    
# Load retriever from the collection
def load_retriever_from_collection(
    collection_name: str,
    search_type: str = "similarity_score_threshold",
    score_threshold: float = 0.3,
    top_k: int = 5
):
    """
    Load a retriever from a Chroma collection with configurable retrieval behavior.

    Args:
        collection_name (str): Name of the Chroma collection.
        search_type (str): Retrieval type (similarity_score_threshold or mmr).
        score_threshold (float): Minimum similarity score for retrieval.
        top_k (int): Number of documents to return.

    Returns:
        Retriever: Configured retriever.
    """

    # Load the persisted collection
    collection = load_chroma_collection(name=collection_name, directory="./persist")
    
    # Build retriever with configurable behavior
    retriever = collection.as_retriever(
        search_type=search_type,
        search_kwargs={
            "score_threshold": score_threshold,
            "k": top_k
        }
    )
    return retriever

# load retirever with metadata filtering
def load_retriever_with_metadata_from_collection(
    collection_name: str,
    search_type: str = "similarity_score_threshold",
    score_threshold: float = 0.3,
    top_k: int = 5,
    metadata_filter: dict = None
):
    """
    Load a retriever from a Chroma collection with configurable retrieval behavior
    and optional metadata filtering.

    Args:
        collection_name (str): Name of the Chroma collection.
        search_type (str): Retrieval type ("similarity_score_threshold" or "mmr").
        score_threshold (float): Minimum similarity score for retrieval.
        top_k (int): Number of documents to return.
        metadata_filter (dict): Optional filter, e.g. {"source": "assets/documents/vacation-policy.pdf"}

    Returns:
        Retriever: Configured retriever.
    """
    collection = load_chroma_collection(name=collection_name, directory="./persist")
    
    retriever = collection.as_retriever(
        search_type=search_type,
        search_kwargs={
            "score_threshold": score_threshold,
            "k": top_k,
            "filter": metadata_filter  # <-- apply metadata filter
        }
    )
    return retriever

# Retrieve with expanded queries
def retrieve_with_expanded_queries(
    collection_name: str,
    queries: List[str],
    search_type: str = "similarity_score_threshold",
    score_threshold: float = 0.3,
    top_k: int = 5,
    metadata_filter: dict = None
) -> List[Document]:
    """
    Retrieve relevant documents from a Chroma collection using one or more expanded queries.

    Args:
        collection_name (str): Name of the Chroma collection.
        queries (List[str]): List of queries, e.g., original query + expanded terms.
        search_type (str): Retrieval type ("similarity_score_threshold" or "mmr").
        score_threshold (float): Minimum similarity score.
        top_k (int): Number of documents to return per query.
        metadata_filter (dict): Optional metadata filter.

    Returns:
        List[Document]: Aggregated, deduplicated documents.
    """
    retriever = load_retriever_from_collection(
        collection_name=collection_name,
        search_type=search_type,
        score_threshold=score_threshold,
        top_k=top_k,
        metadata_filter=metadata_filter
    )
    
    results = []
    for q in queries:
        docs = retriever.get_relevant_documents(q)
        results.extend(docs)
    
    # Deduplicate by source or content
    unique_results = {d.metadata.get("source", d.page_content): d for d in results}
    return list(unique_results.values())

# Simple query expansion function
def expand_query(query: str, n_terms: int = 5) -> list[str]: 
    """ Use LLM to generate related terms for query expansion. """ 
    client = OpenAI() 
    prompt = f""" 
        Generate {n_terms} synonyms of the core word/phrase of the following query for use in document retrieval. 
        Keep them short, noun-phrases. Query: "{query}" """ 
    
    response = client.chat.completions.create( model="gpt-4o-mini", messages=[{"role":"user","content": prompt}], max_tokens=100 ) 
    text = response.choices[0].message.content.strip() 
    
    return [t.strip("-• ") for t in text.split("\n") if t.strip()] 
