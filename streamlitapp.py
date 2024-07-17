import streamlit as st
from PyPDF2 import PdfReader
from typing import List


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def split_text_into_chunks(text: str, chunk_size=500) -> List[str]:
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

def get_triplets(chunks: list, ) -> List[tuple]:
    return []

# Streamlit app
st.title("PDF Content Extractor and Splitter")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    text = extract_text_from_pdf(uploaded_file)

    # Split the text into chunks
    chunks = list(split_text_into_chunks(text))

    # Display the chunks
    st.write("Extracted Text Chunks:")
    for i, chunk in enumerate(chunks):
        st.write(f"Chunk {i + 1}:")
        st.write(chunk)
