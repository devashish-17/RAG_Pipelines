# Import required libraries
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

# Load PDF document
loader = PyPDFLoader('data/Evolution_of_AI.pdf')

# Initialize a text splitter to break the document into chunks of 500 characters with a 100-character overlap
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# Load and split the PDF document into chunks for processing
pages = loader.load_and_split(text_splitter)

# Load environment variables (specifically for the Gemini API key)
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API'))
api_key = os.getenv('GEMINI_API')

# Initialize Google Generative AI embeddings with the provided model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Embed document chunks and store them in Chroma for efficient retrieval
vectordb = Chroma.from_documents(pages, embeddings)

# Set up a retriever with Chroma to search for relevant chunks (top 5 results)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Initialize the large language model for conversational AI
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Define a prompt template to guide AI responses with a specific structure
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
If you don't know, say I don't know
context: {context}
input: {input}
answer:
"""
prompt = PromptTemplate.from_template(template)

# Create a document chain that combines retrieved chunks for more contextual responses
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# Create the final retrieval chain that integrates retrieval and response generation
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Invoke the retrieval chain with different questions and print responses
response = retrieval_chain.invoke({"input": "What is personalised learning environments?"})
print(response["answer"])

response = retrieval_chain.invoke({"input": "What is the use of AI in Education?"})
print(response["answer"])

response = retrieval_chain.invoke({"input": "What is AI?"})
print(response["answer"])
