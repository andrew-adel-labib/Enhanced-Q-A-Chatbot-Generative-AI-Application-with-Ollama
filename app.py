import os 
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, model_name, temperature, num_predict):
    llm = Ollama(model=model_name, temperature=temperature, num_predict=num_predict)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question": question})
    return answer

st.title("Enhanced Q&A Chatbot with Ollama")

llm = st.sidebar.selectbox("Select an Ollama Model: ", ["gemma:2b", "mistral", "llama3:latest"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
num_predict = st.sidebar.slider("Number of Predictions", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask any question!")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, llm, temperature, num_predict)
    st.write(response)

else:
    st.write("Please, Provide the query!")
