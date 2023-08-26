from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv, find_dotenv
import os
from omegaconf import DictConfig
import hydra

load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', None)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)

COLLECTION_NAME = "LibraryFAQ"


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    csv_filepath = 'data/lib-faq-full.csv'
    loader = CSVLoader(file_path=csv_filepath,
                       csv_args={
                           "delimiter":
                           ",",
                           "fieldnames": [
                               'topics', 'question', 'total_views', 'recency',
                               'owner', 'last_updated',
                               'update_since_creation', 'answer', 'created_on'
                           ]
                       })
    documents = loader.load()
    docs = split_docs(documents)
    embeddings = hydra.utils.instantiate(cfg.embeddings)
    persist_dir = cfg.vectorstore.persist_dir

    # db = chromadb.PersistentClient(path=persist_dir)
    vector_store = Chroma.from_documents(documents=docs,
                                         collection_name=COLLECTION_NAME,
                                         embedding=embeddings,
                                         persist_directory=persist_dir)
    vector_store.persist()


if __name__ == "__main__":
    main()
