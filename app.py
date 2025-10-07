from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

load_dotenv(".env")
st.set_page_config(page_title="Ask your Technical Assistant")

@st.cache_resource
def load_knowledge_base(pdf_path="Manual.pdf"):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

st.header("Ask Your PDF")
knowledge_base = load_knowledge_base()  # load once

user_question = st.text_input("Ask a question:")
if user_question:
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
    st.write(response)
