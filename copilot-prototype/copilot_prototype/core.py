import os
# from typing import Any, Dict, List

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import Pinecone
# import pinecone

from pprint import pprint
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key='c4092dfb-b826-4ccc-b82d-288d998c7b98',
    environment='gcp-starter',
)

INDEX_NAME = "compliance-copilot-prototype"


def run_llm(query: str):
    model_name = 'BAAI/bge-large-en-v1.5'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print('*****Pulled PineCone Data*****')
    # chat = ChatOpenAI(
    #     verbose=True,
    #     temperature=0,
    # )
    
    # chat = ChatOllama(model="llama2", verbose=True, temperature=0)

    chat = HuggingFaceHub(repo_id="tiiuae/falcon-7b", huggingfacehub_api_token="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", model_kwargs={'max_length':1000})
    print('***** Model Generated *****')
    qa = RetrievalQA.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    print('***** QA Retriever Built *****')
    return qa({"query": query})

# FOR TESTING IN IDE
if __name__ == "__main__":
    pprint(run_llm(query="What are Ground operations?"))



