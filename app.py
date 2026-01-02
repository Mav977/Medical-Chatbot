from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embedding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
embeddings=download_embedding()
index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":15})
chatModel=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
prompt = PromptTemplate(
    template="""
    "You are an Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def format(docs):
    context="\n\n".join(d.page_content for d in docs )
    return context

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format),
    'question': RunnablePassthrough()
})
parser=StrOutputParser()
chain= parallel_chain | prompt | chatModel | parser

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = chain.invoke(msg)
    print("Response : ", response)
    return str(response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
