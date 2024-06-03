import assemblyai as aai
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import pyttsx3
from io import BytesIO
from st_audiorec import st_audiorec
from typing import IO
import streamlit as st
from langchain_community.document_loaders.assemblyai import TranscriptFormat
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
from openai import OpenAI
from dotenv import load_dotenv
import soundfile as sf
from pydub import AudioSegment

# Load environment variables
load_dotenv()

eleven_api_key = os.getenv("ELEVEN_API_KEY")
aai.settings.api_key = os.getenv("aai.settings.api_key")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY = os.getenv("OPENAI_API_KEY")

# Function to initialize Qdrant client
def qdrant_client():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    qdrant_key = os.getenv("qdrant_key")
    URL = os.getenv("URL")
    client = QdrantClient(
        url=URL,
        api_key=qdrant_key,
    )
    store = Qdrant(client, "my_first_xeven_collection", embedding_model)
    return store

qdrant_store = qdrant_client()

# Function to transcribe audio using Assembly AI
def assembly_ai_voice_to_text(audio_location):
    loader = AssemblyAIAudioTranscriptLoader(file_path=audio_location)
    transcript = loader.load()
    text = transcript[0].page_content
    return text

# Function to transcribe audio using OpenAI Whisper
def transcribe_voice_to_text(audio_location):
    client = OpenAI(api_key=API_KEY)
    with open(audio_location, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text

# Function to generate chat completion using Qdrant and Google Generative AI
def chat_completion_call(text):
    response = qa_ret(qdrant_store, text)
    return response

def text_to_speech_ai(speech_file_path,response):
    client = OpenAI(api_key=API_KEY)
    response = client.audio.speech.create(model="tts-1",voice="nova",input=response)
    response.stream_to_file(speech_file_path)

# Function for text-to-speech conversion using pyttsx3
def text_to_speech_ai_with_elevenlab(speech_file_path, text):
    try:
        engine = pyttsx3.init()
    except (ImportError, RuntimeError):
        st.error("Text-to-speech initialization failed.")
        return

    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    engine.save_to_file(text, 'temp_audio.wav')
    engine.runAndWait()
    audio = AudioSegment.from_wav('temp_audio.wav')
    audio.export(speech_file_path, format="wav")
    return speech_file_path

# Function to handle QA retrieval
def qa_ret(qdrant_store, text):
    try:
        template = """
        You are AI assistant that assists the user by providing answers to their questions by extracting information from the provided context:
        {context} and chat_history if user question is related to chat_history take chat history as context .
        If you do not find any relevant information from context for given question just say ask me another question. You are AI assistant.
        Answer should not be greater than 3 lines.
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever = qdrant_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        output_parser = StrOutputParser()
        rag_chain = setup_and_retrieval | prompt | model | output_parser
        response = rag_chain.invoke(text)
        return response
    except Exception as ex:
        return str(ex)

# Streamlit UI
st.title("QA Retrieval with Speech-to-Text and Text-to-Speech")

st.write("Hi! Click on the voice recorder and let me know how I can help you today.")

method = st.radio("Select Method", ("Openai Whisper + Gemini + OpenAI TTS", "Assembly AI + Gemini + Elevenlabs/Pyttsx"))

wav_audio_data = st_audiorec()
text = st.text_input("Enter your Question here")

if method == "Assembly AI + Gemini + Elevenlabs/Pyttsx" and st.button("Submit"):
    st.write(text)
    api_response = chat_completion_call(text)
    st.write(api_response)
    speech_file_path = 'audio_response.mp3'
    text_to_speech_ai_with_elevenlab(speech_file_path, api_response)
    st.audio(speech_file_path)

if wav_audio_data is not None:
    audio_location = "audio_file.wav"
    with open(audio_location, "wb") as f:
        f.write(wav_audio_data)

    if method == "Openai Whisper + Gemini + OpenAI TTS":
        text = transcribe_voice_to_text(audio_location)
        st.write(text)
        api_response = chat_completion_call(text)
        st.write(api_response)
        speech_file_path = 'audio_response.mp3'
        text_to_speech_ai(speech_file_path, api_response)
        st.audio(speech_file_path)

    if method == "Assembly AI + Gemini + Elevenlabs/Pyttsx":
        text = assembly_ai_voice_to_text(audio_location)
        st.write(text)
        api_response = chat_completion_call(text)
        st.write(api_response)
        speech_file_path = 'audio_response.mp3'
        text_to_speech_ai_with_elevenlab(speech_file_path, api_response)
        st.audio(speech_file_path)

# Display the image with pricing plans
from PIL import Image

image_path = 'img_price.png'
image = Image.open(image_path)
image = image.resize((900, 400))
st.image(image, caption='Pricing plans for audio models')
