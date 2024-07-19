import streamlit as st
from PyPDF2 import PdfReader
from typing import List
from baseline import get_model

@st.cache_resource
def load_model():
    return get_model()

generate_triplets = load_model()

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

# Streamlit app
st.title("PDF Knowledge Graph Generation")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    text = extract_text_from_pdf(uploaded_file)

    # Split the text into chunks
    chunks = list(split_text_into_chunks(text))
    num_chunks = len(chunks)

    # Display the chunks
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Generating triplets for chunk {i + 1}/{num_chunks}..."):
            try:
                triplets = generate_triplets(chunk)

                with st.expander(f"Chunk {i + 1}", expanded=(i == 0)):
                    st.write(f"**Chunk:**")
                    st.write(chunk)


                    st.write("**Triplets:**")
                    st.write(triplets)
                    #for triplet in triplets:
                    #    st.markdown(f"{' margin-top: 6px;' if j > 0 else ''}{' margin-bottom: 12px;' if j == 2 else ''}'>{triplet}", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error while generating triplets for chunk {i + 1}: {str(e)}")