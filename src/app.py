from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from retrievers import generate_embedding_retriever
from dotenv import load_dotenv, find_dotenv
import hydra
from prompts import GENERAL_FAQ_PROMPT
from omegaconf import DictConfig
from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType, Tool, AgentExecutor, BaseSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Union
import gradio as gr
import os
import pinecone

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", None)
PINECONE_ENV = os.environ.get("PINECONE_ENV", None)


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join(
        [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


load_dotenv(find_dotenv())

COLLECTION_NAME = "LibraryFAQ"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def general_faq_agent(cfg: DictConfig) -> None:

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_name = "library-faq"

    llm = hydra.utils.instantiate(cfg.llm)

    # Load vectordb
    embeddings = hydra.utils.instantiate(cfg.embeddings)
    persist_dir = cfg.vectorstore.persist_dir

    # vectordb = Chroma(persist_directory=persist_dir,
    #                   collection_name=COLLECTION_NAME,
    #                   embedding_function=embeddings)

    vectordb = Pinecone.from_existing_index(index_name, embeddings)
    query = "How do I access Lloyd\'s Maritime and Commercial Law Quarterly article?"

    chain_type_kwargs = {'prompt': GENERAL_FAQ_PROMPT}

    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    compression_retriever = generate_embedding_retriever(
        retriever, embeddings, similarity_threshold=0.5)

    general_faq = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs=chain_type_kwargs)

    tools = [
        Tool(name="General FAQ",
             func=general_faq.run,
             description=
             "useful for when you need to answer questions about general \
            library enquiries. Input should be a fully formed question.")
    ]

    agent_kwargs = {
        'extra_prompt_messages': [MessagesPlaceholder(variable_name="memory")]
    }
    memory = ConversationBufferMemory(memory_key="memory",
                                      return_messages=True)
    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             agent_kwargs=agent_kwargs,
                             memory=memory)

    def my_chatbot(input, history):
        history = history or []
        my_history = list(sum(history, ()))
        my_history.append(input)
        my_input = ' '.join(my_history)
        output = agent.run(input)
        history.append((input, output))
        return history, history

    about_me = {
        "TechRead AssistBot":
        "TechRead AssistBot is an AI-powered chatbot designed to provide quick and accurate responses to your inquiries. It employs cutting-edge natural language processing techniques to understand and address your questions effectively.",
        "How It Works":
        "Powered by the GPT-3.5 language model, TechRead AssistBot analyzes your input and generates coherent and contextually relevant replies. It's like having a knowledgeable assistant at your fingertips.",
        "Customization":
        "Tailor the responses to your liking. You can edit the generated replies before sending them, ensuring they align perfectly with your needs.",
        "Data Privacy":
        "Your privacy is paramount. Your interactions with TechRead AssistBot are kept confidential, and we adhere to stringent data security standards.",
        "Accuracy and Improvement":
        "While response accuracy may vary, we're continually refining TechRead AssistBot to enhance its performance and provide you with more accurate information over time.",
    }

    team_members = [
        "Tai Jing Shen",
        "Vivian",
        "Verdio",
        "Gilchris",
    ]

    with gr.Blocks(fn=my_chatbot,
                   inputs="text",
                   outputs="text",
                   title="TechRead AssistBot") as demo:
        gr.Markdown(f"""
            <h1><center><img src="https://logos-download.com/wp-content/uploads/2016/12/Singapore_Management_University_logo_SMU.png" width="200"> TechRead AssistBot</center></h1>
        """)
        chatbot = gr.Chatbot()
        state = gr.State()
        txt = gr.Textbox(
            show_label=False,
            placeholder="Ask me a question and press enter.").style(
                container=False)
        txt.submit(my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
        txt.submit(lambda x: gr.update(value=""), None, [txt], queue=False)
        gr.Markdown("## About Me")
        for idx, (section_title,
                  section_description) in enumerate(about_me.items(), start=1):
            gr.Markdown(f"{section_title}: {section_description}")
        gr.Markdown("## About My Team")
        team_members_list = "\n".join(
            [f"- {member}" for member in team_members])
        gr.Markdown(
            f"We are Go Large or Go Home!, a team of 4 members:\n{team_members_list}"
        )
        gr.Markdown(
            "We're the creators of TechRead AssistBot. If you have any more questions or suggestions, feel free to ask!"
        )

    demo.launch(share=True)


if __name__ == "__main__":
    general_faq_agent()
