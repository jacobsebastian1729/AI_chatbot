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
Answer the question based only on the following context in 20 words:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Convert documents to a string format
def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {"context": retriever | doc2str, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Initialize pygame mixer for audio playback
try:
    pygame.mixer.init()
    pygame_initialized = True
    print("Audio system initialized successfully")
except Exception as e:
    pygame_initialized = False
    print(f"Warning: Audio system initialization failed: {e}")
    print("Will continue with text-only responses")

# Function to get speech input
def listen_for_speech():
    with sr.Microphone() as source:
        print("\nListening for your question...")
        # Adjust the recognizer for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            # Listen for speech
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Processing speech...")
            
            # Recognize speech using Google's speech recognition
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected. Please try again.")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your speech. Please try again.")
            return ""
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return ""
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return ""

# Function to play audio without saving permanent files
def play_audio_response(text):
    if not pygame_initialized:
        print("Audio playback not available")
        return
        
    try:
        # Create a temporary file that will be automatically cleaned up
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_path = temp_file.name
            
        # Generate speech from text
        tts = gTTS(text=text, lang='en')
        tts.save(temp_path)
        
        # Play the audio
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        # Remove the temporary file when done
        try:
            os.unlink(temp_path)
        except:
            pass  # Ignore errors in cleanup
            
    except Exception as e:
        print(f"Error in audio playback: {e}")

# Main interaction loop
print("\n===== Voice-Enabled RAG Chatbot =====")
print("Speak your questions about the document")
print("Say 'exit' or 'quit' to end the session")

while True:
    question = input("\nAsk a question (or type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Exiting...")
        break  # Stop the loop

    response = rag_chain.invoke(question)
    print("\nAnswer:", response)

# Clean up
