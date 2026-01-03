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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
app = Flask(__name__)
load_dotenv()


embeddings=download_embedding()
index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.

class RewrittenQuestion(BaseModel):
    standalone_question: str = Field(
        description="A fully self-contained question rewritten from the conversation"
    )
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":15})
chatModel=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)
MAX_TURNS = 10  # last 10 messages
messages=[]
prompt = PromptTemplate(
    template="""
You are a medical assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Chat history:
{history}

Question:
{question}
""",
    input_variables=["context", "history", "question"]
)
parser=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=RewrittenQuestion)
rewriteQuestion_prompt=PromptTemplate(
    template="""
Given the following chat history and a follow-up question,
rewrite the follow-up question to be a standalone question.

Chat history:
{history}

Follow-up question:
{question}

Standalone question:
\n
{format_instructions}
""",
input_variables=["history", "question"],
partial_variables={"format_instructions": parser2.get_format_instructions()}
)

def format(docs):
    context="\n\n".join(d.page_content for d in docs )
    return context
def return_history(messages):
    history=[]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            history.append(f"AI: {msg.content}")
    return "\n".join(history)

parallel_chain = RunnableParallel({
    "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format),
    "history": RunnableLambda(lambda x: x["history"]),
    "question": RunnableLambda(lambda x: x["question"]),

})



question_reiterator_chain= rewriteQuestion_prompt | chatModel | parser2  | RunnableLambda(lambda x: x.standalone_question)
parallel_chain_for_question_rewrite= RunnableParallel({
    "question": question_reiterator_chain,
    "history":RunnableLambda(lambda x: x["history"])
})
chain= parallel_chain_for_question_rewrite | parallel_chain | prompt | chatModel | parser
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    messages.append(HumanMessage(content=msg))
    
    response = chain.invoke({
        "question":msg,
        "history":return_history(messages)
    })
    messages.append(AIMessage(content=response))
    messages[:] = messages[-MAX_TURNS:]

    print("Response : ", messages)
    return str(response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
