from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from PyPDF2 import PdfReader
from typing import List
from baseline import get_model
from graph_manager import Neo4jTripletManager
from standartize import TripletStandardizer

@st.cache_resource
def load_model():
    return get_model()

generate_triplets, describe_triplets = load_model()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def chunk_text(self, text):
        return self.splitter.split_text(text)
    
chunker = TextChunker(chunk_size=500, chunk_overlap=50)

# Streamlit app
st.title("PDF Knowledge Graph Generation")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    text = extract_text_from_pdf(uploaded_file)

    # Split the text into chunks
    chunks = chunker.chunk_text(text)
    num_chunks = len(chunks)

    standardizer = TripletStandardizer()
    standardizer.init_database("triplet_relations")

    neo4j_manager = Neo4jTripletManager(uri="bolt://localhost:7687", user="neo4j", password="admin")
    neo4j_manager.clear_database()
    
    print(f"Collection count:", standardizer.get_number_of_saved_relations())

    substituitons = []

    # Display the chunks
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Generating triplets for chunk {i + 1}/{num_chunks}..."):
        
            triplets = generate_triplets(chunk)
            descriptions = describe_triplets(chunk, triplets)
            standartized_triplets, subs = standardizer.standartize_triplets(triplets, descriptions)
            substituitons.extend(subs)

            if standartized_triplets is not None:
                triplet_input = [[sub, rel, obj] for sub, (rel, _), obj in standartized_triplets]
                neo4j_manager.add_triplets(triplet_input)

            with st.expander(f"Chunk {i + 1}", expanded=(i == 0)):
                st.write(f"**Chunk:**")
                st.write(chunk)

                st.write("**Triplets:**")
                if not standartized_triplets or len(standartized_triplets) == 0:
                    st.write("No triplets found.")
                else:
                    for triplet in standartized_triplets:
                        st.markdown(f"- **{triplet[0]}** -> **{triplet[1][0]}** -> **{triplet[2]}** | **{triplet[1][0]}** - {(triplet[1][1] if triplet[1][1] is not None else 'N/A')}")
    
    with st.expander(f"Substitutions", expanded=True):
         for subject, (old, new, dist), obj in substituitons:
            st.markdown(f"- **{old}** -> **{new}** in relation **{subject}** -> **{new}**/**{old}** -> **{obj}**")
