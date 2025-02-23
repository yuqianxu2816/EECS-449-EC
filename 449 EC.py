from langchain.llms import Ollama
import base64
import json
import whisper
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate

deepseek_model = Ollama(model="deepseek-r1:1.5b")
qwen_vl_model = Ollama(model="bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh")


memory = ConversationBufferMemory()


memory.save_context({"input": "The player enters the mysterious world"}, {"output": "You find a mysterious path"})


sidekick = ConversationChain(
    llm=deepseek_model,
    memory=memory
)

print("Let's start the game by giving AI the following command:")
print("I want to go to the cyber punk city, give me some suggestions")
print("The AI is going to process the command. Please wait for AI response, it might take time.")
response = sidekick.run("I want to go to the cyber punk city, give me some suggestions")

print("AI says: ", response)


whisper_model = whisper.load_model("base")
print("You can upload voice messages. This is a example for uploading an audio file:")
result = whisper_model.transcribe("I want to go to Atlantis.m4a")  
user_input = result["text"]
print("voice transcription：", user_input)
print("please wait for AI response, it might take time.")
game_response = sidekick.run(user_input)
print("AI says：", game_response)


prompt_template = PromptTemplate(
    input_variables=["history", "question"],
    template="You are exploring in a mysterious world. \n\n{history}\n\nPlayer：{question}\nAI:"
)

game_story = LLMChain(
    llm=deepseek_model,
    prompt=prompt_template,
    memory=memory
)


def game_loop():
    print("Welcome to the mysterious world! You can choose to interact using text or voice in each action.")
    
    while True:
        print("\nChoose input method:")
        print("1. Text Input")
        print("2. Voice Input (Enter your audio file name (e.g. I want to go to the ocean of mystery.m4a) or absolute path if not in the current directory)")
        mode = input("Enter 1 for text, 2 for voice, 'help' for gameplay explanation, or 'exit' to quit: ")
        
        if mode == "exit":
            print("Game ends! Bye!")
            break

        if mode == "help":
            print("Enter 1 for text, 2 for voice, 'help' for gameplay explanation, or 'exit' to quit: ")

        if mode == "1":
            user_input = input("\nYour action: ")
            print("please wait for AI response, it might take time.")
        elif mode == "2":
            audio_file = input("Enter your audio file name (e.g. I want to go to the ocean of mystery.m4a) or absolute path if not in the current directory: ")
            print("please wait for AI response, it might take time.")
            try:
                result = whisper_model.transcribe(audio_file)
                user_input = result["text"]
                print("Voice transcription: ", user_input)
            except Exception as e:
                print("Error processing audio file:", e)
                continue
        else:
            print("Invalid choice. Please select 1 (text) or 2 (voice).")
            continue

        response = game_story.run({"question": user_input})
        
        memory.save_context({"input": user_input}, {"output": response})
        
        print("\nAI says:", response)

game_loop()