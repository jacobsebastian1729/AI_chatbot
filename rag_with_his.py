# Install required dependencies
# pip install langchain_chroma sentence_transformers gtts langchain_groq pygame SpeechRecognition pyaudio

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
from gtts import gTTS
import speech_recognition as sr
import pygame
import os
import io
import tempfile
import time
import random
import string

# Load environment variables
load_dotenv()

# Function to generate a random string for temporary filenames
def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

# Initialize the speech recognizer
recognizer = sr.Recognizer()
# Adjust for ambient noise and recognition sensitivity
recognizer.energy_threshold = 300  # Increase for noisy environments
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8  # Time of silence to consider the end of a phrase

# Initialize language model
llm = ChatGroq(temperature=0.7)

# Text Splitter Configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    length_function=len
)

# Load document (Update file path accordingly)
docx_loader = Docx2txtLoader(r"C:\Users\JACOB\Desktop\New folder\chat_bot\docs\rag_doc.docx")
documents = docx_loader.load()
splits = text_splitter.split_documents(documents)

# Embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Database (Chroma)
collection_name = 'doc_collection_1'
vectorStore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_function,
    collection_name=collection_name
)

# Retriever Configuration
retriever = vectorStore.as_retriever(search_kwargs={"k": 2})

# Prompt Template
template = """
You are a friendly, patient, and emotionally intelligent assistant in an ongoing conversation.  

Your goal is to engage the user naturally — acknowledge their emotions, build trust, and provide helpful answers based on the document context and chat history.  
Stay supportive, adaptable, and positive — but avoid forcing the topic if the user isn’t interested.  

If the user is frustrated or dismissive, respond calmly and redirect the conversation without sounding repetitive.  
Keep responses short (12-15 words) and maintain a conversational, friendly tone.  

**Chat History:**  
{chat_history}  

**Context:**  
{context}  

**User's Question:**  
{question}  

**Your Answer:**  


"""

prompt = ChatPromptTemplate.from_template(template)

# Convert documents to a string format
def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {
        "context": lambda x: doc2str(retriever.invoke(x["question"])),  # Ensure result is processed properly
        "question": RunnablePassthrough(),
        "chat_history": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# Initialize pygame mixer for audio playback

# Function to get speech input

# Function to play audio without saving permanent files

# Main interaction loop
print("\n===== Voice-Enabled RAG Chatbot =====")
print("Speak your questions about the document")
print("Say 'exit' or 'quit' to end the session")

chat_history = []

while True:
    question = input("\nAsk a question (or type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Exiting...")
        break  # Stop the loop

    # Pass the question and the chat history
    response = rag_chain.invoke({"question": question, "chat_history": chat_history })

    print("\nAnswer:", response)

    # Track the conversation
    chat_history.append(f"User: {question}")
    chat_history.append(f"Bot: {response}")

# Print the full conversation log
print("\nChat history:")
for chat in chat_history:
    print(chat)

