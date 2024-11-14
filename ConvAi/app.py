from operator import itemgetter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from chainlit.types import ThreadDict
import chainlit as cl
from prompt import system_instruction  # Import the system instruction
# from llm import chat, messages
from dotenv import load_dotenv
import openai
import tempfile
import os
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO
import time
from elevenlabs import Voice, VoiceSettings, play
from elevenlabs.client import ElevenLabs

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def text_to_speech(text, voice_id='LcfcDJNUP1GQjkzn1xUU'):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    client = ElevenLabs(api_key=api_key)
    voice = Voice(
        voice_id=voice_id,
        settings=VoiceSettings(stability=0.8, similarity_boost=0.75, style=0.0, use_speaker_boost=True),
        language_code="nl"  # Ensure the language code is set to Dutch
    )
    audio_generator = client.generate(text=text, voice=voice, model="eleven_multilingual_v2")
    audio = b"".join(audio_generator)  # Convert generator to bytes
    audio_file = BytesIO(audio)
    return audio_file

def speech_to_text(audio_file):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    client = ElevenLabs(api_key=api_key)
    voices = client.list_voices()
    for voice in voices:
        print(f"Voice ID: {voice.voice_id}, Name: {voice.name}, Language: {voice.language_code}")


def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="nl-NL")
    except sr.UnknownValueError:
        return "Sorry, ik kon dat niet verstaan."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def setup_runnable():
    model = ChatOpenAI(streaming=True, model_name="gpt-3.5-turbo", temperature=0.5)  # Change the model name here
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough()
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)

@cl.on_chat_start
async def on_chat_start():
    setup_runnable()
    opening_message = "Hallo! Ik ben" # for testing
    message = await cl.Message(content=opening_message).send()

    # Convert the opening message to speech and send it as audio
    audio_file = text_to_speech(opening_message)
    audio_data = audio_file.read()
    await cl.Audio(content=audio_data, mime_type="audio/mp3").send(for_id=message.id)  # Use the message ID

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    setup_runnable()

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    audio_file = text_to_speech(res.content)  # Convert response to speech
    audio_data = audio_file.read()
    await cl.Audio(content=audio_data, mime_type="audio/mp3").send(for_id=res.id)  # Send audio to browser

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.step(type="tool")
async def speech_to_text(audio_file):
    client = openai.Client()
       
    response = client.audio.transcriptions.create(
        model="whisper-1",  # Update the model name
        file=audio_file
    )

    return response.text

@cl.on_audio_end
async def on_audio_end(elements):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    
    start_time = time.time()
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)
    end_time = time.time()
    print(f"Transcription took {end_time - start_time} seconds")
    
    user_msg = cl.Message(
        author="You", 
        type="user_message",
        content=transcription
    )
    await user_msg.send()
    await on_message(user_msg)

print("App is running...")
