from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import json
import google.generativeai as genai
from PIL import Image
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

client = MongoClient('localhost', 27017)
db = client['LucknowRestaurants']
collection = db['restaurants']

css = """
<style>
.title {
    font-size: 48px;
    color: #00FFFF;
    text-align: left;
}
.txt {
    font-size: 16px;
    color: #4F8BF9;
    text-align: left;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


def get_gemini_repsonse(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")



st.markdown('<h1 class="title">🧑🏻‍🍳NutriGuide Health Ai</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="txt">Just Upload the image of the food !!</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="txt">NutriGuide is here to give the entire nutrition breakdown of the itme</h1>', unsafe_allow_html=True)

input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me the total calories")

input_prompt = """
just give the name of the dish in one most commanly used word
"""



def fetch_data(collection):
    data = collection.find()
    return list(data)

def convert_to_text(data):
    text_data = []
    for document in data:
        text_data.append(json.dumps(document, default=str))
    return "\n".join(text_data)


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    response = requests.get('https://en.wikipedia.org/wiki/Pasta')
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    s= soup.get_text()


    data = fetch_data(collection)
    text_data = convert_to_text(data)
    print(text_data)

    text = text + s + text_data
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    retriever = vector_store.as_retriever()
    retriever
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()


    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():

    st.header("Queary PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)




    if submit:
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_repsonse(input_prompt, image_data, input)
        st.subheader("The Response is")
        st.write(response)
        print(user_input(f'tell me the name of resturent in lucknow having Basket Chaat'))

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        #url = st.text_input("Enter the url ")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

                st.success("Done")


if __name__ == "__main__":
    main()