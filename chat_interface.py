import streamlit as st
import json
from chat_func import get_api_response
from fuzzywuzzy import fuzz

# Load name mapping data
with open('name_mapping.json', 'r', encoding='utf-8') as f:
    name_mapping = json.load(f)

def display_chat_interface():
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Câu hỏi:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)
            
            if response:
                st.session_state.session_id = response.get('session_id')
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                
                with st.chat_message("assistant"):
                    st.markdown(response['answer'])
                    
                    # Check if the response contains any value from name_mapping
                    for key, value in name_mapping.items():
                        value_without_extension = value.replace('.png', '')
                        if fuzz.partial_ratio(value_without_extension, response['answer']) > 80:  # Adjust the threshold as needed
                            st.session_state.messages.append({"role": "assistant", "content": f"sau đây là ảnh thể hiện {value_without_extension}:"})
                            st.markdown(f"Sau đây là ảnh thể hiện {value_without_extension}:")
                            st.image(f"picture/{value}", use_column_width=True)
                            break
                    
                    with st.expander("Chi tiết"):
                        st.subheader("Câu trả lời")
                        st.code(response['answer'])
                        st.subheader("Model đã dùng")
                        st.code(response['model'])
                        st.subheader("Session ID")
                        st.code(response['session_id'])
            else:
                st.error("Failed to get a response from the API. Please try again.")