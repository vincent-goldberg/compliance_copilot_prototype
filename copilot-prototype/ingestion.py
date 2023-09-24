import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone


pinecone.init(
    api_key='c4092dfb-b826-4ccc-b82d-288d998c7b98',
    environment='gcp-starter',
)

INDEX_NAME = "compliance-copilot-prototype"

doc_path = "/Users/Vincent/Berkeley/w210/compliance_copilot_prototype/compliance_docs/NIST.IR.8270.pdf"

def ingest_docs():
    
    # loading raw compliance doc
    loader = PyPDFLoader(doc_path)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    
    # Spliting it into chunks    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Going to add {len(documents)} to Pinecone")

    # Creating word embeddings
    model_name = 'BAAI/bge-large-en-v1.5'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    

    print('Embeddings created')

    # Uploading to Pinecone
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print('Embeddings uploaded to Pinecone')
    # for doc in documents:
    #     new_url = doc.metadata["source"]
    #     new_url = new_url.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    # embeddings = OpenAIEmbeddings()
    # print(f"Going to add {len(documents)} to Pinecone")
    # Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    # print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    ingest_docs()

