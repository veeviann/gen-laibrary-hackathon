from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter, DocumentCompressorPipeline, LLMChainExtractor)
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.text_splitter import CharacterTextSplitter


def generate_embedding_retriever(retriever,
                                 embeddings,
                                 similarity_threshold: int = 0.5):
    splitter = CharacterTextSplitter(chunk_size=300,
                                     chunk_overlap=0,
                                     separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=similarity_threshold)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever)
    return compression_retriever


def generate_llm_retriever(retriever, llm):
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever)
    return compression_retriever
