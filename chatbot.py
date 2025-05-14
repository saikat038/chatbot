import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere



######################################################################################################################

# Define API Key
COHERE_API_KEY = "NfX96iKqCqTgcJod9s35ashIUboKyXfKDXDvun1Z"

# Creating our UI
# Upload pdf files
st.header("🤖 PDF Chatbot 2.0 (LLM + Reasoning)")

with st.sidebar:
    files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)


######################################################################################################################

# Initialize chat history once
if "history" not in st.session_state:
    st.session_state["history"] = []


# Clear history button
if st.sidebar.button("🧹 Clear History"):
    st.session_state["history"] = []
    st.session_state["submitted"] = []

######################################################################################################################

if files:
    # Combine all PDF text
    text = ""
    
    for file in files:
        # Access the number of pages
        # Extract text from individual pages
        # Inspect metadata, outlines, etc
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()


######################################################################################################################

    # Break the text into chunks
    # rule for breaking it into chunks

    # seperators = "/n" --> break the sentence when a new line comes in
    # chunk_size = 1000 --> 1000 chunks of charecter at a time (tune it based on the output)
    # chunk_overlap= 150 --> keep the last 150 charecter from the last chunk to the present chunk so the meaning in the next chuk remains the same
    # length_function= len --> we are working on length charectaristics
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

######################################################################################################################

    # creating embeddings using openAI API
    # initializing FAISS
    # storing chunks and embeddings

    # Generating embeddings
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")
    vector_store = FAISS.from_texts(chunks, embeddings)


######################################################################################################################


    # callback function to clear the text input
    def clear():
        st.session_state["submitted"] = st.session_state["text"]
        st.session_state["text"] = ""

    user_question = st.text_input("Ask a question about the PDFs", key="text", on_change=clear)

    if st.session_state.get("submitted"):
        st.write(f"You asked: {st.session_state['submitted']}")




        LLM = Cohere(
            cohere_api_key=COHERE_API_KEY,
            temperature=0,
            max_tokens=1000,
            model="command"
        )

        # Search relevant chunks from vector store and passing it to the LLM
        relevant_docs = vector_store.similarity_search(st.session_state["submitted"], k=5)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        You are a highly intelligent assistant that can reason over documents.
        Use the content of the following PDF(s) to answer the user's question.
        Be accurate, concise, and use logic or generate content as needed.
        If the question is not directly answered in the PDF, use your own 
        knowledge and reasoning to respond helpfully.

        PDF Content:
        {context}

        Question:
        {st.session_state["submitted"]}

        Answer:
        """

        response = LLM.predict(prompt)
        st.write(response)

        # store in session history
        st.session_state["history"].append({
            "question": st.session_state["submitted"],
            "answer": response
        })

######################################################################################################################

        # Show chat history
        st.markdown("---")
        st.markdown("## 🕓 Chat History")


        for qa in reversed(st.session_state["history"]):
            st.markdown(f"**Your question:** {qa['question']}")
            st.markdown(f"**Answer:** {qa['answer']}")
            st.markdown("---")
######################################################################################################################
