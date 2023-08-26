from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from langchain.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv, find_dotenv
import os
from omegaconf import DictConfig
import hydra
import pinecone

load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', None)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", None)
PINECONE_ENV = os.environ.get("PINECONE_ENV", None)


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

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_name = "library-faq"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

    csv_filepath = 'data/final/lib-faq-full.csv'
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

    docsearch = Pinecone.from_documents(docs,
                                        embeddings,
                                        index_name=index_name)


if __name__ == "__main__":
    main()
