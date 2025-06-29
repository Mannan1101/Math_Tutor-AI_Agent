import chainlit as cl
from agents import Agent, OpenAIChatCompletionsModel,Runner,set_tracing_disabled, RunConfig
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

set_tracing_disabled(True)
load_dotenv(override=True)

gimini_api_key = os.getenv("GIMINI_API_KEY")
gimini_base_url = os.getenv("GIMINI_BASE_URL")
gimini_model_name = os.getenv("GIMINI_MODEL", "gemini-1.5-flash")



gemini_client = AsyncOpenAI(api_key=gimini_api_key, base_url=gimini_base_url)
gemini_model = OpenAIChatCompletionsModel(openai_client=gemini_client,model=str(gimini_model_name))



math_agent: Agent = Agent(

name = "Math Agent",
instructions= """
you are math agent only solve math questions do not ans other subject questions answers
do no reply if question is not math subject plase 1st be sure this is math questions then reply 
""",

model=gemini_model,
)

@cl.on_chat_start
async def handle_start_chat():
    cl.user_session.set("history", [])
    await cl.Message(content="Math Assistant").send()
   

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content": message.content})
    
    result = await Runner.run(
        math_agent,
        input=history,
        
    )
    
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()
    


