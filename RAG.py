import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

loader = PyPDFLoader('data/Evolution_of_AI.pdf')

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
pages = loader.load_and_split(text_splitter)

load_dotenv()
genai.configure(api_key = os.getenv('GEMINI_API'))
api_key = os.getenv('GEMINI_API')

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Turn the chunks into embeddings and store them in Chroma
vectordb=Chroma.from_documents(pages, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = api_key)

# Create the retrieval chain
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
If you don't know, say I don't know
context: {context}
input: {input}
answer:
"""
prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

response=retrieval_chain.invoke({"input":"What is personalised learning environments?"})
print(response["answer"])

response=retrieval_chain.invoke({"input":"What is the use of AI in Education?"})
print(response["answer"])

response=retrieval_chain.invoke({"input":"What is AI?"})
print(response["answer"])