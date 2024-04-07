from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import config

CHROMA_PATH = "chroma"
DATA_PATH = "Documents"

embedding_function = OllamaEmbeddings(model=config.embeddings_model, base_url=config.base_url)

def generate_data_store():
    """
    Generate the data store by loading documents, splitting text into chunks, and saving to Chroma.
    """
    documents = load_documents()
    print("Loaded documents")
    chunks = split_text(documents)
    print("Split documents into chunks")
    save_to_chroma(chunks)


def load_documents():
    """
    Load documents from a specified directory using a DirectoryLoader instance.

    :return: The loaded documents.
    :rtype: list
    """
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    """
    Split the given list of Document objects into chunks of text.
    Parameters:
        documents (list[Document]): A list of Document objects to be split.

    Returns:
        list[TextChunk]: A list of TextChunk objects containing the split text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    A function to save chunks of documents to Chroma database.

    Parameters:
    chunks (list[Document]): A list of Document objects to be saved to the Chroma database.

    Returns:
    None
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    generate_data_store()