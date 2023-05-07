from typing import Dict, Tuple
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
import time

start = time.time()

def build_index(transcripts: Dict[float, Tuple[float, float, str]]) -> FAISS:
    # Prepare transcript texts as a dictionary
    #documents = {str(start_time): text for start_time, (_, _, text) in transcripts.items()}

    # Split the transcripts into smaller chunks if necessary
    global docs
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    timestamped_transcripts = text_splitter.create_documents(list(transcripts.values()),metadatas=[{'timestamp': k, 'source': index} for index, (k, v) in enumerate(transcripts.items())])
    docs = text_splitter.split_documents(timestamped_transcripts)

    # Create a VectorStore index using the loaded transcript chunks
    '''os.environ["OPENAI_API_KEY"] = "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"
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)'''

    end = time.time()
    print(end-start, "Index building\n")


    return docs

'''def get_timestamps_from_sources(sources: str, docs: list) -> dict:
    # Extract source indices from the sources string
    source_indices = [int(s.strip()) for s in sources.split(',')]

    # Retrieve the timestamps corresponding to the source indices
    timestamps = {}
    for doc in docs:
        metadata = doc.metadata
        source = metadata['source']
        if source in source_indices:
            timestamps[source] = metadata['timestamp']

    return timestamps'''




#def generate_summary(index: LangChain, query: str, max_length: int = 10) -> str:
#    summary = index.generate_sentence(seed_text=query.split(), maxlen=max_length)
#    return ' '.join(summary)