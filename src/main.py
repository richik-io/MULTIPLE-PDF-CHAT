from langchain_community.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from langchain.text_splitter import CharacterTextSplitter
import pickle
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

nltk.download('punkt')

# Streamlit App Interface
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

main_placeholder = st.empty()

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_index.pkl"

if process_url_clicked:
    # Step 1: Load data from the URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    main_placeholder.text("Data Loading...Started...✅✅✅")
    
    # Combine content from all loaded documents
    txt = "\n".join([doc.page_content for doc in data])
    
    # Step 2: Split the text into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    chunks = splitter.split_text(txt)
    main_placeholder.text("Text Splitting...Started...✅✅✅")

    # Step 3: Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the text chunks
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Convert embeddings to numpy arrays for FAISS
    embeddings = np.array(embeddings)

    # Step 4: Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(embeddings)  # Add vectors to the index

    # Step 5: Save the index and chunks for later use
    with open(file_path, 'wb') as f:
        pickle.dump((index, chunks), f)

    main_placeholder.text("FAISS index created and saved...✅✅✅")

# Querying the processed data
query = main_placeholder.text_input("Ask a question about the articles:")

if query:
    # Re-initialize the SentenceTransformer model before query
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 6: Initialize the ChatGroq LLM
    llm = ChatGroq(
        temperature=0.9,
        groq_api_key="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",
        model_name='llama-3.1-70b-versatile'
    )

    # Load the FAISS index and document chunks from the pickle file
    with open(file_path, 'rb') as f:
        index, chunks = pickle.load(f)

    # Step 7: Encode the query using the same SentenceTransformer model
    query_embedding = model.encode([query])

    # Step 8: Search the FAISS index for the nearest neighbors
    D, I = index.search(query_embedding, k=5)  # Retrieve top-5 results

    # Retrieve the top matching chunks
    matching_chunks = [chunks[i] for i in I[0]]

    # Convert the matching chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in matching_chunks]

    # Step 9: Initialize FAISS-based retriever using Langchain's FAISS wrapper
    hf_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Use FAISS from_documents method to build the vector store
    faiss_store = FAISS.from_documents(documents, hf_embeddings)

    # Step 10: Create a retrieval chain using the LLM and FAISS retriever
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_store.as_retriever(),
        return_source_documents=True
    )

    # Query the LLM with the user's input
    result = chain({"query": query})
    
    # Display the result in Streamlit
    st.write(result['result'])