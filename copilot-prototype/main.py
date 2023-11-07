import time
import os
import streamlit as st
from typing import Set
from streamlit_chat import message

from backend.ingestion import ingest_doc
from backend.core import run_llm_summarize, run_llm


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
        loading_message_container = st.empty()  
        loading_message_container.info('Hangtight while I search for the document...', icon="🔎")
        time.sleep(2)
        vectore_store = ingest_doc(file_path, file)
        loading_message_container.empty()
        return True, vectore_store
    return False, None

# Function for uploading and vectorizing document
@st.cache_resource(show_spinner='Processing the document...')
def upload_document_sidebar(file):
    if file_uploaded:
        file_path, file_name = save_upload(file)
        loading_message_container = st.empty()
        loading_message_container.info('Hangtight while I study the document!\nThis can take a few minutes, just enough time to grab a  ☕', 
                               icon="📑")
        time.sleep(5)
        vectore_store = ingest_doc(file_path, file_name)
        loading_message_container.empty() 
        return True, vectore_store
    return False, None



# Sidebar for selecting/uploading document
upload_placeholder = st.empty()

with upload_placeholder.info(" 👈 Select document or upload your own to start chat"):  
    st.sidebar.header("Select a File or Upload New Document")
    with st.sidebar:

        sidebar_completed = False
        st.session_state.vectore_store = None
        
        # Adding empty line for spacing
        st.markdown("") 

         # Radio button for user confirmation with agreement link
        agreement_link = "[User Agreement](https://google.com)"
        user_confirmation = st.checkbox(label=f"I confirm that I have read and understood the {agreement_link}.")
        
        # Adding empty line for spacing
        st.markdown("") 

        if user_confirmation:  # User confirmed, allow document selection/upload
           
            document_selection = st.radio("Would you like to upload a document or select a saved document?",
                             ["Upload", "Select"],
                             captions = ["Load a new document.", "Browse preprocessed documents"])

            # Adding empty line for spacing
            st.markdown("") 

            if document_selection == "Upload":
                # Widget to upload new document
                file_uploaded = st.file_uploader("Upload your PDF file", type="pdf", key='FileUpload')
                upload_sidebar_completed, uploaded_vectore_store = upload_document_sidebar(file_uploaded)

                if file_uploaded:
                    # Successful message
                    message_container = st.empty()
                    message_container.success('Document processed successfully!', icon="✅")
                    st.session_state.message_container = message_container
                    

                    # Changing control variable to enable chatting
                    st.session_state.vectore_store = uploaded_vectore_store
                    sidebar_completed = upload_sidebar_completed
                    # upload_placeholder.empty()
            elif document_selection == "Select":
                # Create a dropdown menu in the sidebar for file selection
                file_selected = st.sidebar.selectbox(label="Select a File", options=saved_docs, placeholder='Choose an option', index=None )
                select_sidebar_completed, selected_vectore_store = select_document_sidebar(file_selected)

                if file_selected:
                    # Successful message
                    message_container = st.empty()
                    message_container.success('Document loaded!', icon="✅")
                    st.session_state.message_container = message_container
                    
                    # # Summarize or chat selection 
                    # col1, col2 = st.columns(2)
                    # with col1:
                    #     summarize = st.button("Summarize")
                    # with col2:
                    #     chat = st.button("Start Chatting")

                    # Changing control variable to enable chatting
                    st.session_state.vectore_store = selected_vectore_store
                    sidebar_completed = select_sidebar_completed
                    # upload_placeholder.empty()
            else:
                st.info("Let's pick a document to review!", icon="☝️")


# Summarize or chat selection 
if sidebar_completed:
    upload_placeholder.info("I've got your document ready! Would you like to chat or get a quick summarization?", icon="👇")
    col1, col2 = st.columns(2)
    with col1:
        summarize = st.button("Summarize")
    with col2:
        chat = st.button("Start Chatting")

    if summarize or chat:
        upload_placeholder.empty()             



# Summarizing document
if "vectore_store" in st.session_state is not None and sidebar_completed and summarize:
    with st.spinner("Generating summary...Now's a great time for ☕ while I type up a thorough report for you!", icon="🧑‍💻"):
        doc_summary = run_llm_summarize()
        st.write(doc_summary)
    

# Starting chat
if "vectore_store" in st.session_state is not None and sidebar_completed and chat:
    vectore_store = st.session_state.vectore_store
    prompt = st.text_input("What's your question?", placeholder="Enter your message here...") or st.button("Submit")

    if prompt:
        with st.spinner("Searching document for the answer..."):
            generated_response, sources = run_llm(vector_database=vectore_store, query=prompt)
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