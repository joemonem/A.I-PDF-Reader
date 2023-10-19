import streamlit as st
from dotenv import load_dotenv


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    st.text_input("Ask a question to your file:")

    with st.sidebar:
        st.subheader("Your files")
        st.file_uploader("Upload files")
        st.button("Process")


if __name__ == "__main__":
    main()
