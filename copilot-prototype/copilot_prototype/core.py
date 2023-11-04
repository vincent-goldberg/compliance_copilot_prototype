from typing import Any



# Chat packages
import torch
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain import HuggingFaceHub

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler





# Global variables
load_dotenv()   # Load environment variables from .env file
huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")
mistral_repo = 'mistralai/Mistral-7B-Instruct-v0.1'


# Tokenizer
embedd_model = 'BAAI/bge-reranker-large'
model_kwargs = {"device": 0}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


# Building LLM

llm = Ollama(model="mistral",
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


def run_llm(vector_database: Any, query: str):
# def run_llm(retriever: Any, query: str, chat_history: List[Dict[str, Any]] = []):
   

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=ollama, retriever=docsearch.as_retriever(), return_source_documents=True
    # )

    retriever = vector_database.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 50})
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    results = qa({"query": query})
    response = results["result"] 
    sources = [doc.metadata["page"] for doc in results["source_documents"]]

    return response, sources


