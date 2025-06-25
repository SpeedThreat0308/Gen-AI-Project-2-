import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with Chat History"
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


#set up Streamlit
st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload PDF's and chat with thier content")

api_key=st.text_input("Provide your Groq API Key:",type="password")

if api_key:
    llm=ChatGroq(model="gemma2-9b-it",groq_api_key=api_key)

    session_id=st.text_input("Session ID:",value="default_session")

    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)


    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            os.makedirs("temp", exist_ok=True)
            pdf_path=os.path.join("temp",uploaded_file.name)
            with open(pdf_path,"wb") as f:
                f.write(uploaded_file.getbuffer())

            loaders=PyPDFLoader(pdf_path)
            documents.extend(loaders.load())
        doc=documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        fin_split=text_splitter.split_documents(doc)
        db=Chroma.from_documents(fin_split,embeddings)
        retriever=db.as_retriever()

        context_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history, Do not answer the question"
            "reformulate it if needed, otherwise return it as it is."
        )
    
        prompt=ChatPromptTemplate.from_messages(
            [
               ("system",context_system_prompt),
               MessagesPlaceholder("chat_history"),
               ("human","{input}")
            ]
        )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,prompt)
       
        #Answer Question prompt
        sys_prompt=(
            "You are an assistant for question answering tasks"
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you don't know"
            "Use three sentences and keep answer concise"
            "\n\n"
            "{context}"
        )
    
        qa_prompt=ChatPromptTemplate.from_messages(
            [
            ("system",sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")        
            ]
        )
    
        doc_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,doc_chain)
    
        def get_session(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
    
        conv_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Enter your Question:")
        if user_input:
            session_history=get_session(session_id)
            response=conv_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable": {"session_id":session_id}
                }
            )

            st.write(st.session_state.store)
            st.write("Assistant:",response["answer"])
            st.write("Chat History:",session_history.messages)

else:
    st.warning("Please Enter the correct Groq API KEY")

    

#Streamlit app





