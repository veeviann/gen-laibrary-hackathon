from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from retrievers import generate_embedding_retriever
from dotenv import load_dotenv, find_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join(
        [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


load_dotenv(find_dotenv())

COLLECTION_NAME = "LibraryFAQ"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    llm = hydra.utils.instantiate(cfg.llm)

    custom_prompt_template = """You are a Librarian chatbot answering questions.
    Use the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    Helpful Answer:"""

    query = "How can I access E-Resources?"

    PROMPT = PromptTemplate(template=custom_prompt_template,
                            input_variables=["context", "question"])
    embeddings = hydra.utils.instantiate(cfg.embeddings)
    persist_dir = cfg.vectorstore.persist_dir

    chain_type_kwargs = {'prompt': PROMPT}
    # ! This might be creating a new Chroma COLLECTION LOL so the code runs
    # ! Even when there's no chroma_db
    vectordb = Chroma(persist_directory=persist_dir,
                      collection_name=COLLECTION_NAME,
                      embedding_function=embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    compression_retriever = generate_embedding_retriever(
        retriever, embeddings, similarity_threshold=0.5)
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=compression_retriever,
                                     chain_type_kwargs=chain_type_kwargs)

    pretty_print_docs(compression_retriever.get_relevant_documents(query))
    result = qa({"query": query})
    print(result)


if __name__ == "__main__":
    main()
