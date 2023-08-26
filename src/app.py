from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
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

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join(
        [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
    
    
load_dotenv(find_dotenv())

COLLECTION_NAME = "LibraryFAQ"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def general_faq_agent(cfg: DictConfig) -> None:

    llm = hydra.utils.instantiate(cfg.llm)

    # Load vectordb
    embeddings = hydra.utils.instantiate(cfg.embeddings)
    persist_dir = cfg.vectorstore.persist_dir
    vectordb = Chroma(persist_directory=persist_dir,
                      collection_name=COLLECTION_NAME,
                      embedding_function=embeddings)

    query = "How do I access Lloyd\'s Maritime and Commercial Law Quarterly article?"

    PROMPT = PromptTemplate(template=GENERAL_FAQ_PROMPT,
                            input_variables=["context", "question"])

    chain_type_kwargs = {'prompt': PROMPT}

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

    with gr.Blocks() as demo:
        gr.Markdown("""<h1><center>Testing Chatbot</center></h1>""")
        chatbot = gr.Chatbot()
        state = gr.State()
        txt = gr.Textbox(
            show_label=False,
            placeholder="Ask me a question and press enter.").style(
                container=False)
        txt.submit(my_chatbot, inputs=[txt, state], outputs=[chatbot, state])

    demo.launch(share=True)


if __name__ == "__main__":
    general_faq_agent()
