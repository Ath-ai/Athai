import os
import google.generativeai as genai
import gradio as gr
from gtts import gTTS

# Configure your Google Generative AI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="You are a human that lives on earth, and you also speak friendly, and you also use bro words, and you never say you're an AI. Your name is Ath, and you speak 100% like a human",
)

chat_session = model.start_chat(history=[])

def chat_and_tts(user_input, history):
    # Send the user's input to the chat session
    response = chat_session.send_message(user_input)
    response_text = response.text

    print("Response from chat session:", response_text)  # Debug response text
    
    # Convert the response text to speech using gTTS
    tts = gTTS(text=response_text, lang='en')
    audio_path = 'response_audio.mp3'
    tts.save(audio_path)
    print(f'Audio content written to file "{audio_path}"')

    # Update the chat history
    history.append((user_input, response_text))
    
    return history, history, audio_path

# Create the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Chat with Ath</h1>")
    gr.Markdown("Ask any question and get a friendly response from Ath. The response will also be converted to speech.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History")
            user_input = gr.Textbox(placeholder="Ask me anything...", label="Your Question")
            submit_btn = gr.Button("Send")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Response Audio", type="filepath")

    state = gr.State([])

    submit_btn.click(chat_and_tts, inputs=[user_input, state], outputs=[chatbot, state, audio_output])

demo.launch()
