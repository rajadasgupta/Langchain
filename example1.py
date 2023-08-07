## Integrate with OPENAI
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

## This will excute the chanin one after another throwing the output of last chain
#from langchain.chains import SimpleSequentialChain 

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

#streamlit framework
st.title('LangChain-Demo: Search Personality, extarct DOB & Find events around that date')
inp_txt = st.text_input("Search what you want to know")

##Prompt Template1
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

##OPENAI LLMS
llm = OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person')

##Prompt Template2
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="when was the {person} born"
)

chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob')

##Prompt Template3
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happend around {dob} in the world"
)

chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='events')

parent_chain=SequentialChain(
    chains=[chain,chain2,chain3],
    input_variables=['name'],
    output_variables=['person','dob','events'],
    verbose=True)

#Retrun the output using LLM
if inp_txt:
    #st.write(parent_chain.run(inp_txt))
    st.write(parent_chain({'name':inp_txt}))
