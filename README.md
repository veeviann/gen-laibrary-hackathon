# gen-laibrary-hackathon

chatbot recsys for gen ai hackathon

## Usage

Set your own API key in testchatbot.py

1. openai.api_key = "sk-xxxx"

1. create new virtual env (if needed)
1. activate new environent via sudo myenv/bin/activate
1. pip install requirements.txt
1. python test_chatbot.py
1. Return two urls (local or gardioLive)

## Rough Outline

### To change the models/embeddings used

1. Refer to the conf folder.
2. Add on the yaml files.
3. Replace the linkage in config.yaml.

### To load to the vectorstore

- Chroma was selected because it is A LOT less buggy. I've tried out the Weaviate, but it keeps giving me error when connecting with LangChain T.T.
- LlamaIndex was used because they can load from the webpage, and by right they are supposed to be better at connecting to the data source. However, the answers from the FAQ page were actually not stored loool, only the questions. Plus I don't know if they split the texts before loading to the DB. If they don't, the query might return long texts. So if we have CSVs and Documents, might want to just use LangChain to load. It's simpler.

```shell
python load_to_vectorstore.py
```

### Linkage to LLM

- Current linkage to LLM is a RetrievalQA. Can change to a "chatbot" using a while loop. But that might not be the best logic for the chatbot.
- Might want to work on the flow on figjam. [Link Here](<https://www.figma.com/file/ZE2B70BE1DFckyuvFUPQfR/Generative-L(AI)brary-Hackathon?type=whiteboard&node-id=58-1316&t=SGPNC3TGULk0nBXr-4>)

```python
while True:
    query = input("")
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])
    chat_history = [(query, result["answer"])]
```

## TODOS

- [ ] Map various data sources and how to load to the vectorstore.
- [ ] Prompt Engineering, but have to decide on the context first.
