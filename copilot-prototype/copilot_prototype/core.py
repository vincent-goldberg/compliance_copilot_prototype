import os
# from typing import Any, Dict, List

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import Pinecone
# import pinecone


from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceHub
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
    # chat = ChatOpenAI(
    #     verbose=True,
    #     temperature=0,
    # )
    chat = HuggingFaceHub(repo_id="gpt2", huggingfacehub_api_token="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")

    qa = RetrievalQA.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"query": query})

if __name__ == "__main__":
    print(run_llm(query="What does NOAA do?"))



