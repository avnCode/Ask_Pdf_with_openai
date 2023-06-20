import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.title('Ask PDF with OpenAI')
menu = ["Home"]
st.sidebar.selectbox("Menu", menu)
pdf_file = st.file_uploader("Upload PDF file", type=['pdf'])

import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"


if pdf_file:
    file_details = {"filename":pdf_file.name,
    "filetype":pdf_file.type,
    "filesize":pdf_file.size}
    st.write(file_details)

    reader = PdfReader(pdf_file)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")


prompt = st.text_input('Plug in your question here')

if prompt:
    query = prompt
    docs = docsearch.similarity_search(query)
    st.write(chain.run(input_documents=docs, question=query))

