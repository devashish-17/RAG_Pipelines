# Import required libraries
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import google.generativeai as genai

# Load environment variables (specifically for the Gemini API key)
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API'))
api_key = os.getenv('GEMINI_API')

# Initialize the large language model for conversational AI
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)


# Load PDF document
def uploadAndSplitPdfFile(f):
    file = 'data/' + f
    loader = PyPDFLoader(file)
    return loader

loader = uploadAndSplitPdfFile('Evolution_of_AI.pdf')


# Create the chunks
def createChunks(loader, size, overlap):
    text_splitter = CharacterTextSplitter(
        chunk_size = size,
        chunk_overlap = overlap,
        length_function = len,
        is_separator_regex = False,
    )    
    # Load & split doc into chunks
    pages = loader.load_and_split(text_splitter)
    return pages

pages = createChunks(loader, 500, 100)


# Generate Embeddings
def generateEmbeddings(model):
    embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    return embeddings

embeddings = generateEmbeddings("models/embedding-001")


vectordb = Chroma.from_documents(pages, embeddings) # storing chuncks in chromaDB

retriever = vectordb.as_retriever(search_kwargs={"k": 5}) # retrieve top 5 results



# Define a prompt template to guide AI responses with a specific structure
def promptTemplate(template_text):
    template = template_text
    prompt = PromptTemplate.from_template(template)

    # Create a document chain that combines retrieved chunks for more contextual responses
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # Create the final retrieval chain that integrates retrieval and response generation
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

text = '''You are a helpful AI assistant. Answer based on the context provided. 
    If you don't know, say I don't know
    context: {context} input: {input} answer:'''

retrieval_chain = promptTemplate(text)



# Write Questions(Prompts)
def getPromptsAndReturnResponse(prompt):
    response = retrieval_chain.invoke({"input": prompt})
    print(response["answer"])    
    
getPromptsAndReturnResponse("What is personalised learning environments?")
getPromptsAndReturnResponse("What is the use of AI in Education?")
getPromptsAndReturnResponse("What is AI?")
