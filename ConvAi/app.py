from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
import chainlit as cl
from prompt import system_instruction  # Import the system instruction
from llm import chat, messages



@cl.on_message
async def main(message: cl.Message):
  messages.append({"role": "user", "content": message.content})
  response = chat(messages)
  messages.append({"role": "system", "content": response})
  

  await cl.Message(
    content=response
  ).send()


@cl.on_chat_start
async def on_chat_start():
  await cl.Message(
    content="Hallo! Ik ben Lotte, jouw digitale tutor. Ik ga je helpen om Nederlands te leren op een leuke en makkelijke manier. We gaan samen oefenen met woorden, zinnen en gesprekjes die je in het dagelijks leven kunt gebruiken, zoals op school of met je vrienden. Als je iets niet begrijpt, maakt dat helemaal niet uit! Ik ben hier om je te helpen en je aan te moedigen. Laten we samen beginnen! "
  ).send()


print()
# @cl.on_chat_start
# async def on_chat_start():
#     model = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_instruction),  # Use the imported system instruction
#             ("human", "{question}"),
#         ]
#     )
#     runnable = prompt | model | StrOutputParser()
#     cl.user_session.set("runnable", runnable)
    
#     # Send a start message
#     await cl.Message(content="Hallo! Ik ben Lotte, je digitale tutor. Hoe kan ik je vandaag helpen?").send()


# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

#     msg = cl.Message(content="")

#     async for chunk in runnable.astream(
#         {"question": message.content},
#         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#     ):
#         await msg.stream_token(chunk)

#     await msg.send()
