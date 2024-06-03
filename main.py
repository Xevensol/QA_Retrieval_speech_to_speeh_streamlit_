import assemblyai as aai
from qdrant_client import QdrantClient
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
load_dotenv()
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")
aai.settings.api_key = os.getenv("aai.settings.api_key")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
api_key = GOOGLE_API_KEY
API_KEY = os.getenv("OPENAI_API_KEY")

def qdrant_client():
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        qdrant_key = os.getenv("qdrant_key")
        URL = os.getenv("URL")
        qdrant_client = QdrantClient(
        url=URL,
        api_key=qdrant_key,
        )
        qdrant_store = Qdrant(qdrant_client,"my_first_xeven_collection" ,embedding_model)
        return qdrant_store

qdrant_store = qdrant_client()


# if not eleven_api_key:
#     raise ValueError("ELEVENLABS_API_KEY environment variable not set")

# client = ElevenLabs(
#     api_key=eleven_api_key,
# )

def assembly_ai_voice_to_text(audio_location):
    loader = AssemblyAIAudioTranscriptLoader(file_path=audio_location)
    transcript = loader.load()
    text = transcript[0].page_content
    return text

def transcribe_voice_to_text(audio_location):
    client = OpenAI(api_key=API_KEY)
    audio_file= open(audio_location, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text

# def chat_completion_call(text):
#     client = OpenAI(api_key=API_KEY)
#     messages = [{"role": "user", "content": text}]
#     response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
#     return response.choices[0].message.content

def chat_completion_call(text):
        response = qa_ret(qdrant_store,text)
        return response
def text_to_speech_ai(speech_file_path,response):
    client = OpenAI(api_key=API_KEY)
    response = client.audio.speech.create(model="tts-1",voice="nova",input=response)
    response.stream_to_file(speech_file_path)

def text_to_speech_ai_with_elevenlab(speech_file_path,text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    engine.save_to_file(text, 'temp_audio.wav')
    engine.runAndWait()
    audio = AudioSegment.from_wav('temp_audio.wav')
    # Export the audio file to the desired path
    audio.export(speech_file_path, format="wav")
    return speech_file_path
def qdrant_client():
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        qdrant_key = os.getenv("qdrant_key")
        URL = os.getenv("URL")
        qdrant_client = QdrantClient(
        url=URL,
        api_key=qdrant_key,
        )
        qdrant_store = Qdrant(qdrant_client,"my_first_xeven_collection" ,embedding_model)
        return qdrant_store

qdrant_store = qdrant_client()



def qa_ret(qdrant_store,text):
    try:
        template = """You are AI assistant that assisant user by providing answer to the question of user by extracting information from provided context:
        {context} and chat_history if user question is related to chat_history take chat history as context .
        if you donot find any relevant information from context for given question just say ask me another quuestion. you are ai assistant.
        Answer should not be greater than 3 lines.
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever= qdrant_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        setup_and_retrieval = RunnableParallel(
                {"context": retriever, "question": RunnablePassthrough()}
                )
            # Load QA Chain
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,google_api_key =api_key)
        output_parser= StrOutputParser()
        rag_chain = (
        setup_and_retrieval
        | prompt
        | model
        | output_parser
        )
        respone=rag_chain.invoke(text)
        return respone
    except Exception as ex:
        return ex



method = st.radio("Select Method",("Openai_wisper_gemini_openai_TTS","Assembly_ai_Gemini_Elevenlab_or_pyttsx"))

st.title("QA Retrieval with speech to text and text to speech")

"""
Hi just click on the voice recorder and let me know how I can help you today ?
"""
wav_audio_data = st_audiorec()
text = st.text_input("Enter your Question here")
if method == "Assembly_ai_Gemini_Elevenlab_or_pyttsx":
    if st.button("submit"):
            st.write(text)
            api_response = chat_completion_call(text)
            st.write(api_response)
            speech_file_path = 'audio_response.mp3'
            text_to_speech_ai_with_elevenlab(speech_file_path, api_response)
            st.audio(speech_file_path)
         


if wav_audio_data is not None:
    # st.audio(wav_audio_data, format='audio/wav')
    ##Save the Recorded File
    audio_location = "audio_file.wav"
    # st.audio(wav_audio_data,format=".wav")
    with open(audio_location, "wb") as f:
        f.write(wav_audio_data)

    

    if method == "Openai_wisper_gemini_openai_TTS":
        text = transcribe_voice_to_text(audio_location)
        st.write(text)
        api_response = chat_completion_call(text)
        st.write(api_response)
        speech_file_path = 'audio_response.mp3'
        text_to_speech_ai(speech_file_path, api_response)
        st.audio(speech_file_path)


    if method == "Assembly_ai_Gemini_Elevenlab_or_pyttsx":
        text = assembly_ai_voice_to_text(audio_location)
        st.write(text)
        api_response = chat_completion_call(text)
        st.write(api_response)
        speech_file_path = 'audio_response.mp3'
        text_to_speech_ai_with_elevenlab(speech_file_path, api_response)
        st.audio(speech_file_path)

from PIL import Image
import streamlit as st

# Correct image path
image_path = 'img_price.png'

# Open and resize the image
image = Image.open(image_path)
image = image.resize((900, 400))  # Adjust the size as needed

# Display the image in Streamlit
st.image(image, caption='Pricing plans for audio models')


             



      




