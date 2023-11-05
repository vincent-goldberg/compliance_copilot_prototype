import time
import os
import streamlit as st
from typing import Set
from streamlit_chat import message

from backend.ingestion import ingest_doc
from backend.core import run_llm






####################
# Utility functions
####################

# Function to list files in upload directory
def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Saving a copy of PDF for vectorization
def save_upload(file):
    file_name = file.name
    
    # Checking if the uploads directory exists, and create it if it doesn't
    outdir = "./backend/uploads/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Checking if the file already exists, and saving it if it doesn't
    file_path = os.path.join(outdir, file_name)
    if not os.path.exists(file_path):        
        # Saving the file
        with open(os.path.join(outdir, file_name), "wb") as f:
            f.write(file.read())

    return file_path, file_name

# Return response sources in formatted string
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"Page: {source}\n"
    return sources_string

####################
# Global Variables
####################

# Creating list of available saved documents
upload_directory = "./backend/uploads/"
saved_docs = list_files(upload_directory)


####################
# Streamlit interface
####################

import streamlit as st
import time

st.title('Welcome to Sibyl 🤖')
st.subheader('Your AI assistant for document review!')

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


# Function for selecting saved document and returning vectorized database
@st.cache_resource(show_spinner='Pulling document from database...')
def select_document_sidebar(file):
    if file_selected:
        outdir = './backend/uploads/'
        file_path = os.path.join(outdir, file)
        message_container = st.empty()  
        message_container.info('Hangtight while I search for the document...', icon="🔎")
        time.sleep(2)
        print('File path: ' + file_path)
        vectore_store = ingest_doc(file_path, file)
        message_container.success('Document loaded! Ready to Chat!', icon="✅")
        return True, vectore_store
    return False, None

# Function for uploading and vectorizing document
@st.cache_resource(show_spinner='Processing the document...')
def upload_document_sidebar(file):
    if file_input:
        file_path, file_name = save_upload(file)
        message_container = st.empty()
        message_container.info('Hangtight while I study the document!\nThis can take a few minutes, just enough time to grab a  ☕', 
                               icon="📑")
        time.sleep(5)
        vectore_store = ingest_doc(file_path, file_name)
        message_container.success('Document processed successfully! Ready to Chat!', icon="✅")         
        return True, vectore_store
    return False, None



# Loading/preparing the document for QA
upload_placeholder = st.empty()

with upload_placeholder.info(" 👈 Select document or upload your own to start chat"):
    st.sidebar.header("Select a File or Upload New Document")
    with st.sidebar:
        # st.subheader('Load Document to Chat')
        
         # Adding empty line for spacing
        st.markdown("") 

        # Create a dropdown menu in the sidebar for file selection
        file_selected = st.sidebar.selectbox(label="Select a File", options=saved_docs, placeholder='Choose an option', index=None )
        
        select_sidebar_completed, selected_vectore_store = select_document_sidebar(file_selected)
        # print("Sidebar selector DB:", vectore_store)
        
        # Adding empty lines for spacing
        st.markdown("") 
        st.markdown("") 
        st.markdown("") 

        # Widget to upload new document
        file_input = st.file_uploader("Upload your PDF file", type="pdf")
        upload_sidebar_completed, uploaded_vectore_store = upload_document_sidebar(file_input)
        
        # If either 
        sidebar_completed = False
        if select_sidebar_completed:
            vectore_store = selected_vectore_store
            sidebar_completed = select_sidebar_completed
            upload_placeholder.empty()
        if upload_sidebar_completed:
            vectore_store = uploaded_vectore_store
            sidebar_completed = upload_sidebar_completed
            upload_placeholder.empty()
        

# Prompt user for question and generating response
if sidebar_completed:
    prompt = st.text_input("What's your question?", placeholder="Enter your message here...") or st.button("Submit")

    if prompt:
        with st.spinner("Searching document for answer..."):
            generated_response, sources = run_llm(vector_database=vectore_store,
                                                  query=prompt)
            formatted_response = (f"{generated_response} \n\n {create_sources_string(set(sources))}")

        st.session_state.chat_history.append((prompt, generated_response))
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(formatted_response)

# Displaying generated response
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(
            user_query,
            is_user=True,
        )
        message(generated_response)
