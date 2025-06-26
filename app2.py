import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
import streamlit as st
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler #Thoughts and action of agent in an interactive streamlit app.

load_dotenv()

#Arxiv Wrapper
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

#Wikipedia Wrapper
wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)


search=DuckDuckGoSearchRun(name="Search")

st.title("🔍 Langchain-Chat using Search via Tools and Agents")
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API KEY:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role":"assistant","content":"Hi I am a chatbot who can search the web. How can I help you?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(model_name="gemma2-9b-it",groq_api_key=api_key,streaming=True)

    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False) #Inserts an invisible container into your app that can be used to hold multiple elements. 
        #This allows you to, for example, insert multiple elements into your app out of order.
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)






