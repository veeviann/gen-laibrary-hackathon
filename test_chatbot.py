import openai
import gradio as gr 

openai.api_key = "sk-xxxx"

start_sequence = "\AI:"
restart_sequence = "\Human:"

prompt = " "

def generate_response(prompt):
    completion = openai.Completion.create(
           model = "text-davinci-003",
           prompt = prompt,
           temperature = 0,
           max_tokens= 500, 
           top_p=1,
           frequency_penalty=0, 
           presence_penalty=0, 
           stop=[" Human:", " AI:"]
       ) 
    return completion.choices[0].text

def my_chatbot(input, history):
    history = history or []
    my_history = list(sum(history, ()))
    my_history.append(input)
    my_input = ' '.join(my_history)
    output = generate_response(my_input)
    history.append((input, output))
    return history, history 

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>Testing Chatbot</center></h1>""")
    chatbot = gr.Chatbot()
    state = gr.State()
    txt = gr.Textbox(show_label=False, placeholder="Ask me a question and press enter.").style(container=False)
    txt.submit(my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
    
demo.launch(share = True)

