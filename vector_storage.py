import os
from typing import List, Dict, Any, Union, Literal
from pinecone import Pinecone
from pydantic import BaseModel, Field

# --- Placeholder Imports (You will replace these with your actual ML pipeline components) ---
# Assuming a generic high-dimensional embedding model for the 1024-dimension index
from langchain_community.embeddings import HuggingFaceBgeEmbeddings # Example for a BGE large model
from langchain_text_splitters import RecursiveCharacterTextSplitter
# -----------------------------------------------------------------------------------------

# --- Configuration ---
# You'd typically load these from environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg")
PINECONE_INDEX_NAME = "assess"
PINECONE_DIMENSION = 1024
# Set your embedding model here
# Replace with the actual model that produces 1024-dim vectors
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5" # Example
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# --- 1. Data Models for Metadata Strategy (Part 1, Requirement 2) ---
# Enforce the required metadata schema for citation and retrieval.

class TextMetadata(BaseModel):
    """Metadata for PDF (text) chunks."""
    source: str = Field(..., description="The filename (e.g., 'history.pdf')")
    type: Literal["text"] = "text"
    page: str = Field(..., description="The page number where the chunk originated (e.g., '12')")

class AudioMetadata(BaseModel):
    """Metadata for Audio Transcript chunks."""
    source: str = Field(..., description="The filename (e.g., 'interview.mp3')")
    type: Literal["audio"] = "audio"
    timestamp: str = Field(..., description="The start timestamp of the segment (e.g., '04:20')")
    # For Part 2, Speaker Diarization:
    speaker_id: str = Field("Unknown", description="The ID of the speaker (e.g., 'HOST' or 'GUEST')")


# --- 2. Conceptual Document Structure (Simulating your pipeline output) ---
# The Audio/Text Ingestion pipelines should output a list of these.

class Document(BaseModel):
    """A unit of content before chunking."""
    page_content: str
    metadata: Union[TextMetadata, AudioMetadata]

# Example of the output from your Audio Ingestion Pipeline (Part 1, Req 1)
def mock_audio_ingestion_output(file_name: str) -> List[Document]:
    return [
        Document(
            page_content="The architect mentioned 'green spaces' as a new urban planning concept.",
            metadata=AudioMetadata(source=file_name, timestamp="00:00:15", speaker_id="GUEST")
        ),
        Document(
            page_content="That's a key difference from the earlier zoning laws, I agree.",
            metadata=AudioMetadata(source=file_name, timestamp="00:00:25", speaker_id="HOST")
        )
    ]

# Example of the output from your PDF Ingestion Pipeline
def mock_pdf_ingestion_output(file_name: str) -> List[Document]:
    return [
        Document(
            page_content="The foundation of the zoning laws was established in the mid-1970s, which linked housing to industrial development.",
            metadata=TextMetadata(source=file_name, page="4")
        ),
        Document(
            page_content="The subsequent chapter discusses the role of local media archives in historical research.",
            metadata=TextMetadata(source=file_name, page="5")
        )
    ]


# --- 3. Unified Ingestion and Upsert Logic ---

def ingest_documents_to_pinecone(
    pinecone_index: Pinecone,
    audio_docs: List[Document],
    text_docs: List[Document],
    embedding_model
) -> Dict[str, int]:
    """
    Chunks, embeds, and upserts the documents (Audio & Text) into the Pinecone index.
    """

    # 1. Initialize Text Splitter for uniform chunking
    # This splitter preserves metadata from the parent document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_documents = audio_docs + text_docs
    total_chunks = 0

    print(f"--- Preparing to process {len(all_documents)} documents. ---")

    # Pinecone upsert typically works best in batches (e.g., 100 vectors per batch)
    BATCH_SIZE = 100
    upsert_vectors = []

    for doc in all_documents:
        # Create a dictionary for the initial document to work with LangChain's splitter
        lc_doc = {"page_content": doc.page_content, "metadata": dict(doc.metadata)}

        # Split the document content
        chunks: List[str] = text_splitter.split_text(doc.page_content)

        # Preserve metadata for each chunk
        for i, chunk in enumerate(chunks):
            # The metadata for the chunk is the same as the parent document's metadata
            chunk_metadata = dict(doc.metadata)

            # Create a unique ID for the vector
            # ID format: {source_file}_{type}_{page/timestamp_id}_{chunk_index}
            # This ensures traceability (Observability - Part 3)
            if chunk_metadata['type'] == 'audio':
                unique_id_part = chunk_metadata['timestamp'].replace(':', '-')
            else: # type == 'text'
                unique_id_part = f"page-{chunk_metadata['page']}"

            vector_id = f"{chunk_metadata['source']}_{chunk_metadata['type']}_{unique_id_part}_{i}"

            # Add the text content to the metadata for observability/debugging retrieval
            chunk_metadata['text'] = chunk

            # Generate the embedding (Conceptual step - replace with your model's call)
            # embeddings = embedding_model.embed_documents([chunk])[0] # For a list of chunks
            # For this example, we'll embed one by one or use a placeholder
            embeddings = embedding_model.embed_query(chunk)


            # Prepare the vector for upsert
            upsert_vectors.append((vector_id, embeddings, chunk_metadata))
            total_chunks += 1

            # Upsert in batches
            if len(upsert_vectors) >= BATCH_SIZE:
                pinecone_index.upsert(vectors=upsert_vectors)
                print(f"Upserted {len(upsert_vectors)} chunks. Total chunks: {total_chunks}")
                upsert_vectors = []

    # Upsert the final batch
    if upsert_vectors:
        pinecone_index.upsert(vectors=upsert_vectors)
        print(f"Upserted final batch of {len(upsert_vectors)} chunks. Total chunks: {total_chunks}")

    return {"total_chunks_upserted": total_chunks}


# --- 4. Main Execution Block ---

def run_pinecone_ingestion_pipeline(audio_file_path: str, pdf_file_path: str):
    """Main function to run the full ingestion pipeline."""

    # 1. Initialize Pinecone connection
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Correctly extract index names from the list_indexes() response
        if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
             raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")

        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")

    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        # Graceful Degradation (Part 3)
        return

    # 2. Initialize the Embedding Model
    # NOTE: You MUST use a model that outputs 1024-dimensional vectors.
    try:
        # Conceptual placeholder for a 1024-dim model (e.g., BGE-large)
        embedding_model = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"✅ Initialized Embedding Model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"❌ Error initializing embedding model: {e}")
        # Graceful Degradation (Part 3)
        return

    # 3. Ingest Data (Simulate your Part 1, Req 1 & PDF loading)
    print("--- Starting Data Ingestion Simulation ---")
    audio_docs = mock_audio_ingestion_output(audio_file_path)
    text_docs = mock_pdf_ingestion_output(pdf_file_path)
    print(f"Loaded {len(audio_docs)} audio segments and {len(text_docs)} text segments.")

    # 4. Process and Upsert
    print("--- Starting Chunking, Embedding, and Upsert to Pinecone ---")
    try:
        result = ingest_documents_to_pinecone(
            pinecone_index=pinecone_index,
            audio_docs=audio_docs,
            text_docs=text_docs,
            embedding_model=embedding_model
        )
        print(f"✨ Ingestion complete. {result['total_chunks_upserted']} chunks upserted.")
    except Exception as e:
        print(f"❌ An error occurred during upsert: {e}")
        # Graceful Degradation (Part 3)


if __name__ == '__main__':
    # Replace these with your actual mock file paths
    DRIVE_PATH = "/content/drive/MyDrive/assess"
    MOCK_AUDIO_FILE = os.path.join(DRIVE_PATH, "audio-sample-1..mp3")
    MOCK_PDF_FILE = os.path.join(DRIVE_PATH, "thepublicdomain1.pdf")

    # Ensure you have your API Key set in your environment
    run_pinecone_ingestion_pipeline(MOCK_AUDIO_FILE, MOCK_PDF_FILE)
