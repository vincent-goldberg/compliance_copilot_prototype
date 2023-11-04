import time
import os
import streamlit as st
from streamlit_chat import message

from copilot_prototype.ingestion import ingest_doc
from copilot_prototype.core import run_llm






####################
# Utility functions
####################

# Saving a copy of PDF for vectorization
def save_upload(file):
    file_name = file.name
    
    # Checking if the uploads directory exists, and create it if it doesn't
    outdir = "./copilot_prototype/uploads/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Checking if the file already exists, and saving it if it doesn't
    file_path = os.path.join(outdir, file_name)
    if not os.path.exists(file_path):        
        # Saving the file
        with open(os.path.join(outdir, file_name), "wb") as f:
            f.write(file.read())

    return file_path, file_name

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



# Uploading and vectorizing document
@st.cache_resource()
def load_document_sidebar(file):
    if file_input:
        file_path, file_name = save_upload(file)
        vectore_store = ingest_doc(file_path, file_name)
        # vectore_store = ingest_doc(file_path, file_name)
        st.success('Document uploaded successfully!', icon="✅")
        with st.spinner(text='Reviewing the document before we begin chatting!'):
            time.sleep(5)
            st.success('Ready to Chat!', icon="✅")
        
        return True, vectore_store
    return False, None




upload_placeholder = st.empty()
with upload_placeholder.info(" 👈 Upload document to start chat"):
    with st.sidebar:
        st.subheader('Load Document to Chat')
        file_input = st.file_uploader("Upload your PDF file", type="pdf")
        sidebar_completed, vectore_store = load_document_sidebar(file_input)
        if sidebar_completed:
            upload_placeholder.empty()


if sidebar_completed:
    prompt = st.text_input("What's your question?", placeholder="Enter your message here...") or st.button("Submit")

    if prompt:
        with st.spinner("Generating response..."):
            generated_response, sources = run_llm(vector_database=vectore_store,
                                                  query=prompt)
            
            # st.write(generated_response)

            # generated_response = "This is you AI generated response"
            # sources = [1,2,3]

            formatted_response = (
            f"{generated_response} \n\n {sources}"
        )
            st.write(formatted_response)

        st.session_state.chat_history.append((prompt, generated_response))
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append((formatted_response))


#             generated_response, sources = run_llm(
#                 retriever=retriever,
#                 query=prompt,
#                 chat_history=st.session_state["chat_history"]
#             )

#             sources = set(
#                 [doc.metadata["source"] for doc in generated_response["source_documents"]]
#             )
#             formatted_response = (
#                 f"{generated_response['answer']} \n\n {sources}"
#             )

#             st.session_state.chat_history.append((prompt, generated_response["answer"]))
#             st.session_state.user_prompt_history.append(prompt)
#             st.session_state.chat_answers_history.append(formatted_response)


# if st.session_state["chat_answers_history"]:
#     for generated_response, user_query in zip(
#         st.session_state["chat_answers_history"],
#         st.session_state["user_prompt_history"],
#     ):
#         message(
#             user_query,
#             is_user=True,
#         )
#         message(generated_response)




# st.title('Welcome to Sibyl', )
# st.subheader('Your AI assistant for document review!')

# upload_placeholder = st.empty()

# with upload_placeholder.info(" 👈 Upload document to start chat"):
#     with st.sidebar:
#         st.subheader('Load Document to Chat')
#         file_input = st.file_uploader("Upload your PDF file", type="pdf")
#         if file_input:
#             save_upload(file_input)
#             # st.write('Document upload successfully!')
#             st.success('Document uploaded successfully!', icon="✅")

#             with st.spinner(text='Reviewing the document before we begin chatting!'):
#                 time.sleep(5)
#                 st.success('Ready to Chat!', icon="✅")                
#             upload_placeholder.empty()
#         st.stop()

# st.rerun()
# prompt = st.text_input("What's your question?", placeholder="Enter your message here...") or st.button(
#     "Submit"
# )


# if prompt:
#     with st.spinner("Generating response..."):
#     #   generated_response, sources = run_llm(query=prompt)
#         st.write('working on it')
#     #   st.write(generated_response)
