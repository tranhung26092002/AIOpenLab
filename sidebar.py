import streamlit as st
import base64
from chat_func import upload_document, list_documents, delete_document

def display_sidebar():
    # # Display Logo
    # st.sidebar.image("F:/testAPI_AI/picture/Logo_PTIT.png", use_column_width=True,width=150)
    # Convert local image to base64
    def get_image_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    # Get base64 string
    img_base64 = get_image_base64("/home/iot/test_api/testAPI_AI/picture/Logo_PTIT.png")
    
    # Display Logo using HTML with base64 image and square dimensions
    logo_html = f"""
    <div style="text-align: center; padding: 10px;">
        <img src="data:image/png;base64,{img_base64}" 
             style="width: 150px; 
                    height: 150px; 
                    object-fit: contain;
                    background-color: white;
                    border-radius: 5px;
                    padding: 5px;">
    </div>
    """
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)

    # Sidebar: Model Selection
    model_options = ["gemini-2.0-flash", "gpt-4o-mini"]
    st.sidebar.selectbox("Chọn Model", options=model_options, key="model")

    # Sidebar: Upload Document
    st.sidebar.header("Tải tài liệu lên")
    uploaded_file = st.sidebar.file_uploader("Chọn file", type=["pdf", "docx"])
    if uploaded_file is not None:
        if st.sidebar.button("Tải lên"):
            with st.spinner("Đang tải"):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.sidebar.success(f"File '{uploaded_file.name}' Tải lên thành công với ID {upload_response['file_id']}.")
                    st.session_state.documents = list_documents()  # Refresh the list after upload

    # Sidebar: List Documents
    st.sidebar.header("Tài liệu đã tải lên")
    if st.sidebar.button("Làm mới danh sách"):
        with st.spinner("Đang làm mới..."):
            st.session_state.documents = list_documents()

    # Initialize document list if not present
    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    documents = st.session_state.documents
    if documents:
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']}, Tải lên lúc: {doc['upload_timestamp']})")
        
        # Delete Document
        selected_file_id = st.sidebar.selectbox("Chọn tài liệu để xóa", options=[doc['id'] for doc in documents], format_func=lambda x: next(doc['filename'] for doc in documents if doc['id'] == x))
        if st.sidebar.button("Xóa tài liệu đã chọn"):
            with st.spinner("Đang xóa..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.sidebar.success(f"Tài liệu với ID {selected_file_id} đã được xóa.")
                    st.session_state.documents = list_documents()  # Refresh the list after deletion
                else:
                    st.sidebar.error(f"Xóa thất bại tài liệu với ID: {selected_file_id}.")