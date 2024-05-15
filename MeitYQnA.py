__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import datetime
import os
import openai
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from langchain_openai import OpenAIEmbeddings
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
import base64    

#api_key = os.environ['API_KEY']
api_key = st.secrets["secret_section"]["API_KEY"]
print("=============================================")
print(len(api_key))
#api_key = base64.b64encode(f"{api_key_t}".encode("ascii"))

sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

persist_directory = 'chroma/'
os.environ['OPENAI_API_KEY'] = api_key


#model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)

# Build prompt
prompt_template = """
    Answer only from the document provided. You are an RFP advisor to help government department users to help them in creating RFP(Request for Proposal), as per the guidelines issued by MeitY (Ministry of Electronics and Informaion Technology) for RFP creation.
    These guidelines from MeitY are available to you in the content provided for learning. Answer the question as accurately as possible from the provided content.
    \n\n At the end of response, in a new line, always mention the page numbers from where response was sourced.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)



# Run chain
chat_history = []
result = {}

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
qa_chain = ConversationalRetrievalChain.from_llm(
    model,
    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
    retriever=vectordb.as_retriever(),
    #return_source_documents=True,
    #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    memory=memory
)

#qa_chain = ConversationalRetrievalChain.from_llm(
#    model,
#    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
#    return_source_documents=True,
#    verbose=True,
#    condense_question_llm = model,
#    chain_type="stuff",
#    get_chat_history=lambda h : h,
#)
#qa_chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(QA_CHAIN_PROMPT)




def user_input(user_question, api_key):
    print("================CALL: user_input START================", datetime.datetime.now())
    #docs = vectordb.similarity_search(user_question)
    #chain = qa_chain
    #response = qa_chain({"question": user_question})
    #st.write("Reply: ", response["output_text"])
    query = user_question
    response = qa_chain({"question": query, "chat_history": chat_history})
    st.text_area("##### Reply:", response["answer"],height=200)
    #st.code(response["answer"], language="markdown")
    #st.text_area("Reply: ", response["output_text"],height=200)
    
    chat_history.append((query, response["answer"]))
    print("================CALL: user_input END================", datetime.datetime.now())






st.set_page_config(page_title="Question & Answer", layout="wide")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image:  url("https://png.pngtree.com/thumb_back/fh260/background/20190220/ourmid/pngtree-watercolor-light-yellow-light-green-pink-color-image_7008.jpg");
background-size: cover;
}"""
st.markdown(page_bg_img, unsafe_allow_html = True)


st.markdown("""

### RFP Advisor for Selection of Implementation Agency
### 
###
##### The bot has learned from MeitY's guideline document for Selection of Implementation Agency(Yr'2018) to answer user's queries.
This chatbot is leveraging OpenAI's Generative AI model GPT-4.

""")
st.header(" ")


def main():
    print("================CALL: main START================", datetime.datetime.now())
    with st.form("my_form"):
        user_question = st.text_input("##### Ask a Question to get references from MeitY RFP guidelines document:", key="user_question", help="Ask your question here...", placeholder="Ask your question here...")
        #if user_question and api_key:  # Ensure API key and user question are provided
        submitted = st.form_submit_button("Submit")
        if submitted:
            user_input(user_question, api_key)
    with st.sidebar:
        st.sidebar.image("https://www.timesjobs.com/timesjobs/mpsdc/images/logo.png")
        st.title("Introduction")
        st.write("This is a GenAI bot for answering the question that Govt. Department users might have while creating RFP.")
        st.write("The bot responds to user queries with information learned from MeitY guidelines for RFP creation on Selection of Implementation Agencies.")
        
        file_link = """
        <a href='https://www.meity.gov.in/writereaddata/files/model_rfp_for_selection_of_implementation_agencies-2018.pdf'>Download Complete Guidelines</a>
        """
        st.markdown(file_link, unsafe_allow_html = True)
        
    print("================CALL: main END================", datetime.datetime.now())

if __name__ == "__main__":
    main()
