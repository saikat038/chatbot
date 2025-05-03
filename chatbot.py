'''
Required Libraries
pip install
streamlit --> for creating UI interfaces
pypdf2    --> allows us to read pdf files
langchain --> an interface to use openAI services
faiss-cpu/gpu --> Vector store to store embeddings

'''

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Cohere




# defining openAI API key
COHERE_API_KEY = "bm24O1eFEWEuFHYHDYJufVIGkMxcgcvS5SmQsc24"






# Creating our UI
# Upload pdf files
st.header("Chatbot")

with st.sidebar:
    st.title("Your Document")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

######################################################################################################################


# Extract the text from the pdf
if file is not None:
    
    # reading the pdf page by page
    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        # extracting text from every page
        text += page.extract_text()
        # st.write(text)


######################################################################################################################

    # Break the text into chunks
    # rule for breaking it into chunks

    # seperators = "/n" --> break the sentence when a new line comes in
    # chunk_size = 1000 --> 1000 chunks of charecter at a time (tune it based on the output)
    # chunk_overlap= 150 --> keep the last 150 charecter from the last chunk to the present chunk so the meaning in the next chuk remains the same
    # length_function= len --> we are working on length charectaristics

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap= 150,
        length_function= len
    )

    # passing the text into text splitter
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

######################################################################################################################
   
   

    # - creating embeddings using openAI API
    # - initializing FAISS
    # - storing chunks and embeddings

    # Generating embeddings
    embeddings = CohereEmbeddings(cohere_api_key= COHERE_API_KEY, model="embed-english-v3.0")

    # Creating vector store -- FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)


######################################################################################################################
    # callback function to clear the text input
    def clear():
        st.session_state["submitted"] = st.session_state["text"]
        st.session_state["text"] = ""

    # Get user's question
    user_question = st.text_input("Ask anything", key="text", on_change=clear)
    
    # Do similerity search (return the chunk which matches the question)
    if st.session_state.get("submitted"):
        st.write(f"You have asked: {st.session_state['submitted']}\n")
        match = vector_store.similarity_search(st.session_state["submitted"])

######################################################################################################################

        # Output results

        # take the question
        # get relevent document
        # pass it to LLM
        # generate the output

        # defining LLM
        # we are doing fine tuning here
        # defining temperature (0-1) (lower the value specific the ans, higher the temperature vague the ans will be)
        # max token defines the limit of the response
        LLM = Cohere(
            cohere_api_key = COHERE_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model= "command"
        )

        # load_qa_chain takes the LLM parameters and type
        chain = load_qa_chain(llm = LLM, chain_type="stuff")
        # chain.run takes the chunks and the question as an input
        response = chain.run(input_documents= match ,question = user_question)
        st.write(response)