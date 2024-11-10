from operator import itemgetter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
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


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def text_to_speech(text, lang='nl'):
    tts = gTTS(text=text, lang=lang)
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

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
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)

@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("chat_history", [{"role": "system", "content": system_instruction}])
    setup_runnable()
    opening_message = (
        "Hallo! Ik ben Lotte, jouw digitale tutor. Ik ga je helpen om Nederlands te leren op een leuke en makkelijke manier. "
        "We gaan samen oefenen met woorden, zinnen en gesprekjes die je in het dagelijks leven kunt gebruiken, zoals op school of met je vrienden. "
        "Als je iets niet begrijpt, maakt dat helemaal niet uit! Ik ben hier om je te helpen en je aan te moedigen. Laten we samen beginnen! "
        "Ik ben niet een echte leraar, maar een AI-assistent die je helpt om Nederlands te leren. Ik herinner je er af en toe aan dat ik een AI-assistent ben, zodat je niet in de war raakt."
    )
    message = await cl.Message(content=opening_message).send()

    # Convert the opening message to speech and send it as audio
    audio_file = text_to_speech(opening_message)
    audio_data = audio_file.read()
    await cl.Audio(content=audio_data, mime_type="audio/mp3").send(for_id=message.id)  # Use the message ID

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    chat_history = [{"role": "system", "content": system_instruction}]

    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
            chat_history.append({"role": "user", "content": message["output"]})
        else:
            memory.chat_memory.add_ai_message(message["output"])
            chat_history.append({"role": "assistant", "content": message["output"]})

    cl.user_session.set("memory", memory)
    cl.user_session.set("chat_history", chat_history)
    setup_runnable()

@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    chat_history = cl.user_session.get("chat_history")
    runnable = cl.user_session.get("runnable")  # type: Runnable

    chat_history.append({"role": "user", "content": message.content})

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)
    chat_history.append({"role": "assistant", "content": res.content})

    audio_file = text_to_speech(res.content)  # Convert response to speech
    audio_data = audio_file.read()
    await cl.Audio(content=audio_data, mime_type="audio/mp3").send(for_id=res.id)  # Send audio to browser

    cl.user_session.set("chat_history", chat_history)

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
