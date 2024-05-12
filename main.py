import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]

# Configure your embedding model and vector store
embedding = OpenAIEmbeddings()

astra_db_store = AstraDBVectorStore(
    collection_name="PTT_unstructured",
    embedding=embedding,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)
print("Astra vector store configured")

prompt = """
Answer the question based only on the engineering documents supplied as context. If you don't know the answer, say "I don't know".
Context: {context}
Question: {question}
Your answer:
"""

llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=False, temperature=0)

chain = (
    {"context": astra_db_store.as_retriever(), "question": RunnablePassthrough()}
    | PromptTemplate.from_template(prompt)
    | llm
    | StrOutputParser()
)

def get_response(question):
    return chain.invoke(question)

def main():
    st.title('PTT Q&A Assistant')

    user_input = st.text_input("Ask your question:", "What are the maximum allowable values for carbon equivalent (CE) and the chemical composition of pressure-retaining materials?")
    if st.button('Submit'):
        response = get_response(user_input)
        st.text_area("Response:", value=response, height=300)

if __name__ == '__main__':
    main()
