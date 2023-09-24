from typing import Set

from copilot_prototype.core import run_llm
import streamlit as st
from streamlit_chat import message


# def create_sources_string(source_urls: Set[str]) -> str:
#     if not source_urls:
#         return ""
#     sources_list = list(source_urls)
#     sources_list.sort()
#     sources_string = "sources:\n"
#     for i, source in enumerate(sources_list):
#         sources_string += f"{i+1}. {source}\n"
#     return sources_string


st.header("LangChain🦜🔗 Compliance Copilot")

prompt = st.text_input("Prompt", placeholder="Enter your message here...") or st.button(
    "Submit"
)

# response = 'This is a test response!'

# st.text(response)

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt
        )


        response = generated_response['result']
        st.text(response)
        
        # sources = set(
        #     [doc.metadata["source"] for doc in generated_response["source_documents"]]
        # )
        # formatted_response = (
        #     f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        # )

#         st.session_state.chat_history.append((prompt, generated_response["answer"]))
#         st.session_state.user_prompt_history.append(prompt)
#         st.session_state.chat_answers_history.append(formatted_response)

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