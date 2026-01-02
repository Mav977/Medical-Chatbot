from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
def load_docs(data):
    loader=DirectoryLoader(
        data,
        glob="**/*pdf",
        loader_cls=PyPDFLoader
    )
    docs=loader.lazy_load()
    return docs

def filter_doc(docs: List[Document]) -> List[Document]: 
    minimal_docs : List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src, "page":doc.metadata.get("page")}
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk=text_splitter.split_documents(minimal_docs)
    return text_chunk


def download_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
    return embeddings
    