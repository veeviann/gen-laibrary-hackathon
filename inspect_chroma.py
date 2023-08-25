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

    print(collection.peek())


if __name__ == "__main__":
    main()
