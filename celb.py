from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain, LLMChain
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define Prompt Templates
first_prompt = PromptTemplate(input_variables=['name'], template="Tell me about celebrity {name}")
second_prompt = PromptTemplate(input_variables=['name'], template="What are the achievements of {name}?")
third_prompt=PromptTemplate(input_variables=['dob'],template=" give me major events in that {dob} across the world")
# Set up Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descrption_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')


# Define LLM
llm = Ollama(model="llama2")
chain1=LLMChain(llm=llm,prompt=first_prompt,verbose=True,output_key='person',memory=person_memory)
chain2=LLMChain(llm=llm,prompt=second_prompt,verbose=True,output_key='dob',memory=dob_memory)
chain3=LLMChain(llm=llm,prompt=third_prompt,verbose=True,output_key='description',memory=descrption_memory)

# Sequential Chain Setup
celb_chain = SequentialChain(
    chains=[chain1,chain2,chain3],
    input_variables=['name'],
    output_variables=['person','dob','description'],
    verbose=True
)

# Streamlit UI
st.title("Celebrity Information Chatbot")

input_text = st.text_input("Enter the name of the celebrity:")
if input_text:
    st.write(celb_chain({'name':input_text}))
    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descrption_memory.buffer)
