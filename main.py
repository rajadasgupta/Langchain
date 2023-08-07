## Integrate with OPENAI
import imp
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

#streamlit framework
st.title('LangChain demo FOR CELEBRITY SEARCH')
inp_txt = st.text_input("Search what you want to know")

##Prompt Template
first_input_prompt = PromptTemplate(
input_variables=['name'],
template_format="Tell me about celebrity {name}"
)

##OPENAI LLMS
llm = OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True)

if inp_txt:
    st.write(chain.run(inp_txt))

