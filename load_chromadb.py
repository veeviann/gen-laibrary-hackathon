from llama_index.vector_stores import ChromaVectorStore
from llama_index import StorageContext, load_index_from_storage

import chromadb
import hydra
from omegaconf import DictConfig

COLLECTION_NAME = "LibraryFAQ"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    persist_dir = cfg.vectorstore.persist_dir

    db = chromadb.PersistentClient(path=persist_dir)
    embeddings = hydra.utils.instantiate(cfg.embeddings)
    collection = db.get_collection(COLLECTION_NAME,
                                   embedding_function=embeddings)
    # ! Keeps getting the sqlite3 error 
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = load_index_from_storage(vector_store)
    query_engine = index.as_query_engine()
    response = query_engine.query('How do I access e-resources?')
    print(response)


if __name__ == "__main__":
    main()
