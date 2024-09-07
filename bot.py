from langchain_openai import ChatOpenAI
from langchain_core.prompts import  ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import  streamlit as st
import os 
from dotenv import load_dotenv
import getpass

load_dotenv()  ## intilizing it 

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## prompt template

prompt=ChatPromptTemplate.from_messages([("system","you are working as helpful assistant.Respond to the questions what the user asks"),
                                          ("user","Question:{question}")])   ## user prompt


## Stramlit framework

st.title('Chatbot demo using langchain and ollama')
input_text=st.text_input("Search the topic what u want")

## Calling the llms

llm=Ollama(model="llama2")
out_parser=StrOutputParser()
chain=prompt|llm|out_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))


