import os
import re
import string
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()  # Load environment variables from .env

class PDFProcessor:
    """
    Processes PDF documents:
    - Extracts raw text
    - Splits text into sections based on headings
    - Cleans text for NLP
    - Splits text into chunks with metadata
    """
    def __init__(self, pdf_folder: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    def extract_text(self, file_path: str) -> str:
        """Extracts raw text from a PDF using PyPDF2."""
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def split_into_sections(self, text: str) -> dict:
        """Splits text into sections based on detected headings."""
        lines = text.splitlines()
        sections = {}
        current_heading = "Document"
        current_content = []

        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped.split()) <= 6 and stripped[0].isupper() and not stripped.endswith("."):
                if current_content:
                    sections[current_heading] = " ".join(current_content).strip()
                current_heading = stripped
                current_content = []
            else:
                current_content.append(stripped)

        if current_content:
            sections[current_heading] = " ".join(current_content).strip()

        return sections

    def clean_text(self, text: str, lowercase: bool = True) -> str:
        """Cleans text: lowercases, removes punctuation, normalizes spaces."""
        if lowercase:
            text = text.lower()
        text = text.replace("\n", " ").replace("\t", " ")
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process_pdfs(self):
        """
        Processes all PDFs in the folder:
        - Extracts text
        - Splits into sections
        - Splits sections into chunks
        - Returns chunks and metadata
        """
        all_chunks = []
        all_metadatas = []

        for file in self.files:
            pdf_path = os.path.join(self.pdf_folder, file)
            raw_text = self.extract_text(pdf_path)
            sections = self.split_into_sections(raw_text)

            for section_title, content in sections.items():
                cleaned_content = self.clean_text(content)
                chunks = self.text_splitter.split_text(cleaned_content)
                all_chunks.extend(chunks)
                all_metadatas.extend([{"source": file, "section": section_title}] * len(chunks))

        return all_chunks, all_metadatas

class VectorStoreManager:
    def __init__(self, pdf_folder: str, persist_dir: str, embeddings, chunks = None, metadatas = None):
        """
        Manages a Chroma vector store for a collection of PDF documents.

        Responsibilities:
        -----------------
        1. Checks if the vector store is up-to-date based on the last modified 
        time of the PDFs in the folder.
        2. Recreates the vector store only if any PDF has been updated, avoiding 
        unnecessary recomputation.
        3. Persists the vector store along with a timestamp of the last update.
        4. Loads the existing vector store if it is already up-to-date.
        5. Provides easy access to the vector store for similarity search or RAG pipelines.

        Attributes:
        -----------------
        pdf_folder: folder containing PDFs
        persist_dir: folder to persist Chroma vector store
        embeddings: OpenAIEmbeddings instance
        chunks: list of preprocessed text chunks
        metadatas: list of metadata dictionaries for each chunk
        """
        self.pdf_folder = pdf_folder
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.chunks = chunks
        self.metadatas = metadatas
        self.timestamp_file = os.path.join(persist_dir, "last_updated.json")
        self.vectorstore = None

    def save_timestamp(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        now = datetime.now()
        with open(self.timestamp_file, "w") as f:
            json.dump({"last_updated": now.isoformat()}, f)
        return now

    def load_timestamp(self):
        if not os.path.exists(self.timestamp_file):
            return None
        with open(self.timestamp_file, "r") as f:
            data = json.load(f)
            return datetime.fromisoformat(data["last_updated"])

    def is_up_to_date(self):
        last_updated = self.load_timestamp()
        if last_updated is None:
            return False
        
        for file in os.listdir(self.pdf_folder):
            if not file.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(self.pdf_folder, file)
            if datetime.fromtimestamp(os.path.getmtime(file_path)) > last_updated:
                return False
        return True

    def load_or_create(self):
        """
        Loads existing vector store if up to date, otherwise recreates it.
        Returns the vectorstore instance.
        """
        last_updated = self.load_timestamp()
        if not self.is_up_to_date():
            print("Vector store is outdated. Recreating...")
            self.vectorstore = Chroma.from_texts(
                texts=self.chunks,
                embedding=self.embeddings,
                metadatas=self.metadatas,
                persist_directory=self.persist_dir
            )
            self.vectorstore.persist()
            last_updated = self.save_timestamp()
            print(f"Vector store recreated. Last updated: {last_updated}")
        else:
            print(f"Vector store is up to date. Last updated: {last_updated}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        return self.vectorstore