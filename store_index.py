from dotenv import load_dotenv
import os
load_dotenv()
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_docs, text_split, filter_doc, download_embedding
extracted_data=load_docs(data='data/')
filter_data = filter_doc(extracted_data)
text_chunk=text_split(filter_data)

embedding = download_embedding()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    docsearch=PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
    )

