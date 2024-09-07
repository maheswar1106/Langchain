from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain.chains import SequentialChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt templates
first_prompt = ChatPromptTemplate.from_template(
    "Tell me about doctor {name}"
)

second_prompt = ChatPromptTemplate.from_template(
    "What was {person}'s specialization?"
)

# Memory setups
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
specialization_memory = ConversationBufferMemory(input_key='person', memory_key='specialization_history')

# LLMs
llm = Ollama(model="llama2")

# Chains
first_chain = LLMChain(
    llm=llm,
    prompt=first_prompt,
    verbose=True,
    output_key='person',
    memory=person_memory
)

second_chain = LLMChain(
    llm=llm,
    prompt=second_prompt,
    verbose=True,
    output_key='specialization',
    memory=specialization_memory
)

# Sequential chain setup
chain = SequentialChain(
    chains=[first_chain, second_chain],
    input_variables=['name'],
    output_variables=['specialization']
)

# Streamlit setup
st.title("Doctor Information Chatbot")

input_text = st.text_input("Ask about the doctor:")

if input_text:
    response = chain.run({'name': input_text})
    st.write(response)
