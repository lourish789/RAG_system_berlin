import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv

# Third-party libraries (Ensure these are in your requirements.txt)
from flask import Flask, request, jsonify
from pinecone import Pinecone, Index
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
# Conceptual dependencies (need mock or external implementation)
# For the full test, you would replace these with actual whisper/pypdf/text-splitter/etc. logic
from typing import NamedTuple

# --- 1. Configuration and Logging ---
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. Data Models and Conceptual Pipeline Components ---

@dataclass
class DocumentChunk:
    """A single unit of content for the vector store"""
    id: str
    text: str
    source: str
    type: str # 'audio' or 'text'
    metadata: Dict[str, Union[str, int, float]]
    embedding: Optional[List[float]] = None # Will be populated during embedding step

def generate_chunk_id(content_snippet: str, source_file: str, index: int) -> str:
    """Generates a simple unique ID for a chunk"""
    # Simple hash based ID to meet the ID requirement (Production would use UUID or more robust hash)
    import hashlib
    content_hash = hashlib.sha256(content_snippet.encode('utf-8')).hexdigest()[:8]
    return f"{source_file.replace('.', '_')}_{content_hash}_{index}"

@dataclass
class QueryResult:
    """Structured query result with citations"""
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    timestamp: str
    query_metadata: Dict[str, Any]

# --- CONCEPTUAL/MOCK PIPELINE COMPONENTS ---

class EmbeddingService:
    """MOCK: Handles embedding generation for the pipeline."""
    def __init__(self, model_name: str):
        logger.info(f"Loading MOCK Embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback for conceptual use if model cannot be loaded
            self.dimension = 1024 
            logger.warning(f"Could not load SentenceTransformer. Using mock dimension: {self.dimension}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # MOCK IMPLEMENTATION: Replace with actual bulk embedding call
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        # For a deployable script, you must ensure SentenceTransformer is used correctly
        # or use a cloud-based embedding service.
        return self.model.encode(texts, convert_to_tensor=False).tolist()

class AudioProcessor:
    """MOCK: Handles audio transcription and segmentation (e.g., Whisper)"""
    def __init__(self, model_size: str):
        logger.info(f"Initialized MOCK Audio Processor (Whisper size: {model_size})")

    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict[str, Any]]:
        # MOCK IMPLEMENTATION: Replace with actual Whisper/diarization output
        logger.warning(f"Using MOCK transcript for {Path(audio_path).name}. You must implement Whisper/Diarization.")
        return [
            {
                'text': "The architect mentioned 'green spaces' as a new urban planning concept, emphasizing sustainability.",
                'timestamp': "00:00:15",
                'start_time': 15.0,
                'end_time': 25.0
            },
            {
                'text': "Success is primarily defined by the achievement of long-term community resilience, not short-term profit.",
                'timestamp': "00:01:30",
                'start_time': 90.0,
                'end_time': 105.0
            }
        ]

class PDFProcessor:
    """MOCK: Handles PDF text extraction and chunking (e.g., pypdf + text splitter)"""
    def __init__(self):
        logger.info("Initialized MOCK PDF Processor")

    def extract_text_with_pages(self, pdf_path: str, chunk_size: int) -> List[Dict[str, Any]]:
        # MOCK IMPLEMENTATION: Replace with actual PDF parsing and text splitting
        logger.warning(f"Using MOCK PDF chunks for {Path(pdf_path).name}. You must implement pypdf/text-splitter.")
        return [
            {
                'text': "The foundation of urban zoning laws was established in the mid-1970s, which directly linked housing to industrial development policies.",
                'page': "4",
                'chunk_index': 0
            },
            {
                'text': "Chapter 5 discusses the role of local media archives in historical research, stating that preservation is key to defining a successful legacy.",
                'page': "5",
                'chunk_index': 0
            }
        ]

class PineconeVectorStore:
    """Vector store operations (Pinecone)"""
    def __init__(self, api_key: str, index_name: str, dimension: int):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        
        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self._connect_or_create_index()
            logger.info(f"Connected to Pinecone index: {index_name} (Dim: {dimension})")
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise RuntimeError(f"Failed to connect to Pinecone: {e}")

    def _connect_or_create_index(self):
        if self.index_name not in self.pc.list_indexes().names:
            logger.warning(f"Index {self.index_name} not found. Creating it...")
            self.pc.create_index(
                name=self.index_name, 
                dimension=self.dimension,
                metric='cosine' # Use a standard metric
            )
        return self.pc.Index(self.index_name)

    def upsert_chunks(self, chunks: List[DocumentChunk]):
        """Upsert chunks into Pinecone in batches"""
        vectors = []
        for chunk in chunks:
            # Prepare metadata for filtering (Part 2, Module A)
            metadata = {
                'source': chunk.source,
                'type': chunk.type,
                'text': chunk.text, # Observability (Part 3)
                **chunk.metadata 
            }
            # Ensure embedding is available
            if chunk.embedding is None:
                logger.error(f"Chunk {chunk.id} has no embedding.")
                continue
                
            vectors.append((chunk.id, chunk.embedding, metadata))

        # Batch upsert
        if vectors:
            self.index.upsert(vectors=vectors, batch_size=100)
            logger.info(f"Upserted {len(vectors)} vectors in total.")

    def get_stats(self):
        """Get index statistics"""
        return self.index.describe_index_stats()

# --- 3. Attribution Engine (Part 1, Requirement 3) ---

class AttributionEngine:
    """
    Core query engine that ensures proper attribution and citation
    """

    def __init__(
        self,
        pinecone_api_key: str,
        gemini_api_key: str,
        index_name: str = 'assess',
        embedding_model: str = 'BAAI/bge-large-en-v1.5'
    ):
        """Initialize the attribution engine with all required services"""
        logger.info("Initializing Attribution Engine")

        try:
            # Initialize embedding model (used for query embedding only)
            self.embedding_service = EmbeddingService(model_name=embedding_model)

            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")

            # Initialize Gemini
            genai.configure(api_key=gemini_api_key)
            self.llm = genai.GenerativeModel('gemini-pro')
            logger.info("Initialized Gemini model")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            # Graceful Degradation (Part 3)
            raise

    def query_archive(
        self,
        question: str,
        top_k: int = 5,
        filter_by_type: Optional[str] = None,
        filter_by_source: Optional[str] = None,
        include_raw_chunks: bool = True
    ) -> QueryResult:
        """
        Main query interface with proper attribution (The Attribution Engine)

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            filter_by_type: Filter by 'audio' or 'text'
            filter_by_source: Filter by specific source file
            include_raw_chunks: Include raw retrieved chunks in result

        Returns:
            QueryResult with answer and citations
        """
        logger.info(f"Processing query: {question}")
        query_start = datetime.now()

        try:
            # Step 1: Embed the query
            query_embedding = self._embed_query(question)
            logger.info("Query embedded successfully")

            # Step 2: Build filters (Part 2, Module A - Metadata Filtering)
            filter_dict = self._build_filters(filter_by_type, filter_by_source)

            # Step 3: Retrieve relevant chunks
            retrieved_chunks = self._retrieve_chunks(
                query_embedding,
                top_k,
                filter_dict
            )
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

            if not retrieved_chunks:
                logger.warning("No relevant chunks found")
                return self._create_empty_result(question)

            # Step 4: Generate answer with citations
            answer_data = self._generate_attributed_answer(question, retrieved_chunks)

            # Step 5: Create structured result (Observability - Part 3)
            query_end = datetime.now()
            query_duration = (query_end - query_start).total_seconds()

            result = QueryResult(
                question=question,
                answer=answer_data['answer'],
                citations=answer_data['citations'],
                retrieved_chunks=retrieved_chunks if include_raw_chunks else [],
                timestamp=query_end.isoformat(),
                query_metadata={
                    'duration_seconds': query_duration,
                    'chunks_retrieved': len(retrieved_chunks),
                    'filters_applied': filter_dict,
                    'top_k': top_k
                }
            )

            logger.info(f"Query completed in {query_duration:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            # Graceful Degradation (Part 3)
            return self._create_empty_result(question, f"A system error occurred: {str(e)}")


    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            embedding = self.embedding_service.model.encode(query, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise RuntimeError(f"Failed to embed query: {e}")

    def _build_filters(
        self,
        filter_by_type: Optional[str],
        filter_by_source: Optional[str]
    ) -> Optional[Dict]:
        """Build Pinecone metadata filters"""
        # This structure allows for future expansion into Hybrid Search/BM25 (Part 2, Module A)
        filters = {}

        if filter_by_type:
            filters['type'] = {'$eq': filter_by_type}

        if filter_by_source:
            filters['source'] = {'$eq': filter_by_source}
            
        return filters if filters else None

    def _retrieve_chunks(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_dict: Optional[Dict]
    ) -> List[Dict]:
        """Retrieve chunks from vector store with error handling"""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            return results.get('matches', [])

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Graceful Degradation (Part 3)
            raise RuntimeError(f"Failed to retrieve chunks: {e}")

    def _generate_attributed_answer(
        self,
        question: str,
        chunks: List[Dict]
    ) -> Dict[str, Any]:
        """Generate answer with strict citation requirements"""
        try:
            context = self._format_context_for_prompt(chunks)
            prompt = self._create_attribution_prompt(question, context)

            response = self.llm.generate_content(prompt)
            answer_text = response.text

            citations = self._extract_citations_from_chunks(chunks)

            return {
                'answer': answer_text,
                'citations': citations
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Graceful degradation (Part 3)
            return {
                'answer': f"I encountered an error generating the answer based on the retrieved context: {str(e)}",
                'citations': []
            }

    def _format_context_for_prompt(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string for the LLM"""
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            text = metadata.get('text', 'No content available')
            source = metadata.get('source', 'unknown')
            chunk_type = metadata.get('type', 'unknown')

            # Citation location logic
            location_info = ""
            if chunk_type == 'audio':
                timestamp = metadata.get('timestamp', 'unknown')
                location_info = f"at timestamp {timestamp}"
            else: # type == 'text'
                page = metadata.get('page', 'unknown')
                location_info = f"on page {page}"

            formatted_chunks.append(
                f"[Chunk {i}]\n"
                f"Source: {source} ({chunk_type}) {location_info}\n"
                f"Content: {text}\n"
            )

        return "\n---\n".join(formatted_chunks)

    def _create_attribution_prompt(self, question: str, context: str) -> str:
        """Create prompt with strict citation requirements"""
        # Prompt includes the strict rules to enforce the Citation Requirement (Part 1, Req 3)
        return f"""You are a professional archivist assistant at the Berlin Media Archive. Your role is to help researchers by providing accurate, well-cited answers from historical documents and audio interviews.\n\nRETRIEVED CONTEXT:\n{context}\n\nRESEARCHER'S QUESTION:\n{question}\n\nCRITICAL INSTRUCTIONS:\n1. Answer ONLY using information from the provided context above (Faithfulness check - Part 2, Module C)\n2. Every factual claim MUST include a citation\n3. Citation format:\n   - For audio: "(Source: [filename] at [timestamp])"\n   - For text: "(Source: [filename], Page [number])"\n4. If multiple sources support a point, cite all of them\n5. If the context doesn't contain enough information to answer the question, explicitly state: "The available archives do not contain sufficient information to answer this question."\n6. Do not make assumptions or add information not present in the context\n\nNow provide your answer with proper citations:"""

    def _extract_citations_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Extract structured citation information"""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            citation = {
                'citation_id': i,
                'source': metadata.get('source', 'unknown'),
                'type': metadata.get('type', 'unknown'),
                'relevance_score': round(chunk.get('score', 0.0), 4)
            }
            if metadata.get('type') == 'audio':
                citation['timestamp'] = metadata.get('timestamp', 'unknown')
            else:
                citation['page'] = metadata.get('page', 'unknown')
            citation['text_snippet'] = metadata.get('text', '')[:200] + "..."
            citations.append(citation)
        return citations

    def _create_empty_result(self, question: str, error_message: str = "I could not find any relevant information in the archives to answer this question.") -> QueryResult:
        """Create result when no chunks are found or on error (Error Handling - Part 3)"""
        return QueryResult(
            question=question,
            answer=error_message,
            citations=[],
            retrieved_chunks=[],
            timestamp=datetime.now().isoformat(),
            query_metadata={
                'duration_seconds': 0,
                'chunks_retrieved': 0,
                'filters_applied': None,
                'top_k': 0
            }
        )

# --- 4. Archive Pipeline (Orchestration for Ingestion) ---

class ArchivePipeline:
    """Orchestrates the complete RAG pipeline, focused on Ingestion for setup."""

    def __init__(self, config: dict):
        self.config = config
        self.embedding_service = EmbeddingService(model_name=self.config.get('embedding_model', 'BAAI/bge-large-en-v1.5'))
        self.audio_processor = AudioProcessor(model_size=self.config.get('whisper_model', 'base'))
        self.pdf_processor = PDFProcessor()
        self.vector_store = PineconeVectorStore(
            api_key=self.config['pinecone_api_key'],
            index_name=self.config.get('index_name', 'assess'),
            dimension=self.embedding_service.dimension
        )

    def ingest_documents(self, audio_path: str = None, pdf_path: str = None):
        """Ingest documents into the vector store."""
        logger.info("STARTING DOCUMENT INGESTION")
        all_chunks = []

        # Process audio (Audio Ingestion Pipeline - Part 1, Req 1)
        if audio_path and os.path.exists(audio_path):
            try:
                segments = self.audio_processor.transcribe_with_timestamps(audio_path)
                filename = Path(audio_path).name
                audio_chunks = [
                    DocumentChunk(
                        id=generate_chunk_id(seg['text'], filename, i),
                        text=seg['text'],
                        source=filename,
                        type='audio',
                        metadata={'timestamp': seg['timestamp'], 'start_time': seg['start_time'], 'end_time': seg['end_time']}
                    ) for i, seg in enumerate(segments)
                ]
                all_chunks.extend(audio_chunks)
                logger.info(f"Audio processed: {len(audio_chunks)} chunks created")
            except Exception as e:
                logger.error(f"Audio processing failed: {e}. Degrading gracefully.") # Graceful Degradation (Part 3)

        # Process PDF
        if pdf_path and os.path.exists(pdf_path):
            try:
                pdf_chunks_raw = self.pdf_processor.extract_text_with_pages(
                    pdf_path, chunk_size=self.config.get('chunk_size', 500)
                )
                filename = Path(pdf_path).name
                pdf_chunks = [
                    DocumentChunk(
                        id=generate_chunk_id(data['text'], filename, i),






#--- 4. Archive Pipeline (Orchestration for Ingestion) ---

class ArchivePipeline:
    """Orchestrates the complete RAG pipeline, focused on Ingestion for setup."""

    def __init__(self, config: dict, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.audio_processor = AudioProcessor(model_size=self.config.get('whisper_model', 'base'))
        self.pdf_processor = PDFProcessor()
        self.vector_store = PineconeVectorStore(
            api_key=self.config['pinecone_api_key'],
            index_name=self.config.get('index_name', 'assess'),
            dimension=self.embedding_service.dimension
        )

    def ingest_documents(self, audio_path: str = None, pdf_path: str = None):
        """Ingest documents into the unified vector store."""
        logger.info("STARTING DOCUMENT INGESTION")
        all_chunks = []

        # Process audio (Part 1, Req 1)
        if audio_path and os.path.exists(audio_path):
            try:
                segments = self.audio_processor.transcribe_with_timestamps(audio_path)
                filename = Path(audio_path).name
                audio_chunks = [
                    DocumentChunk(
                        id=generate_chunk_id(seg['text'], filename, i),
                        text=seg['text'],
                        source=filename,
                        type='audio',
                        metadata={
                            'timestamp': seg['timestamp'], 
                            'start_time': seg['start_time'], 
                            'end_time': seg['end_time'],
                            'speaker_id': seg.get('speaker_id', 'Unknown')
                        } # Speaker Diarization (Module B)
                    ) for i, seg in enumerate(segments)
                ]
                all_chunks.extend(audio_chunks)
                logger.info(f"Audio processed: {len(audio_chunks)} chunks created")
            except Exception as e:
                logger.error(f"Audio processing failed: {e}. Degrading gracefully.") 

        # Process PDF
        if pdf_path and os.path.exists(pdf_path):
            try:
                pdf_chunks_raw = self.pdf_processor.extract_text_with_pages(
                    pdf_path, chunk_size=self.config.get('chunk_size', 500)
                )
                filename = Path(pdf_path).name
                pdf_chunks = [
                    DocumentChunk(
                        id=generate_chunk_id(data['text'], filename, i),
                        text=data['text'],
                        source=filename,
                        type='text',
                        metadata={'page': data['page'], 'chunk_index': data['chunk_index']}
                    ) for i, data in enumerate(pdf_chunks_raw)
                ]
                all_chunks.extend(pdf_chunks)
                logger.info(f"PDF processed: {len(pdf_chunks)} chunks created")
            except Exception as e:
                logger.error(f"PDF processing failed: {e}. Degrading gracefully.") 


        if not all_chunks:
            logger.error("No documents were successfully processed!")
            return False

        # Generate embeddings and upload (Unified Vector Store - Part 1, Req 2)
        all_texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedding_service.embed_batch(all_texts)
        
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding

        self.vector_store.upsert_chunks(all_chunks)
        logger.info("INGESTION COMPLETE")
        return True


# --- 5. Flask Backend (Render Deployment) ---
#-- 5. Flask Backend (Render Deployment) ---

# Get config from environment
CONFIG = {
    'pinecone_api_key': os.getenv('PINECONE_API_KEY', "YOUR_PINECONE_KEY"),
    'gemini_api_key': os.getenv('GEMINI_API_KEY', "YOUR_GEMINI_KEY"),
    'index_name': os.getenv('PINECONE_INDEX_NAME', 'assess'),
    'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5'),
    'whisper_model': os.getenv('WHISPER_MODEL', 'base'),
    'chunk_size': int(os.getenv('CHUNK_SIZE', '500')),
    # Define placeholder file paths for local/ingestion testing
    'audio_file': os.getenv('AUDIO_FILE', 'mock_interview.mp3'), 
    'pdf_file': os.getenv('PDF_FILE', 'mock_history.pdf')
}

app = Flask(__name__)

# Initialize Global Services
try:
    # Initialize Embedding Service once for both Attribution Engine and Pipeline
    GLOBAL_EMBEDDING_SERVICE = EmbeddingService(model_name=CONFIG['embedding_model'])

    ATTRIBUTION_ENGINE = AttributionEngine(
        pinecone_api_key=CONFIG['pinecone_api_key'],
        gemini_api_key=CONFIG['gemini_api_key'],
        index_name=CONFIG['index_name'],
        embedding_model=CONFIG['embedding_model']
    )
    # Pass the shared embedding service instance to the pipeline
    PIPELINE = ArchivePipeline(CONFIG, GLOBAL_EMBEDDING_SERVICE) 
    logger.info("RAG Services initialized for Flask app.")
except Exception as e:
    logger.error(f"FATAL: Could not initialize RAG services. API will not function: {e}")
    ATTRIBUTION_ENGINE = None
    PIPELINE = None


@app.route('/ingest', methods=['POST'])
def ingest_data():
    """Endpoint to run the full ingestion pipeline."""
    if not PIPELINE:
        return jsonify({"error": "Pipeline not initialized. Check server logs."}), 500
    
    data = request.get_json(silent=True) or {}
    # Allow specifying files via JSON, otherwise use defaults
    audio_path = data.get('audio_path', CONFIG['audio_file'])
    pdf_path = data.get('pdf_path', CONFIG['pdf_file'])

    try:
        success = PIPELINE.ingest_documents(audio_path=audio_path, pdf_path=pdf_path)
        if success:
            stats = PIPELINE.vector_store.get_stats()
            return jsonify({
                "status": "Ingestion Complete",
                "message": "Documents processed, embedded, and upserted successfully.",
                "total_vectors": stats.get('total_vector_count')
            }), 200
        else:
            return jsonify({"status": "Ingestion Failed", "message": "Check server logs for specific file errors."}), 500
    except Exception as e:
        logger.error(f"Ingestion API failed: {e}")
        return jsonify({"status": "Ingestion Failed", "error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_archive_api():
    """Endpoint for the Attribution Engine's query_archive function."""
    if not ATTRIBUTION_ENGINE:
        return jsonify({"error": "Attribution Engine not initialized. Check server logs."}), 500
        
    data = request.get_json(silent=True)
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    question = data['question']
    top_k = data.get('top_k', 5)
    filter_type = data.get('filter_by_type')
    filter_source = data.get('filter_by_source')
    filter_speaker = data.get('filter_by_speaker') # Module B Filter

    try:
        result = ATTRIBUTION_ENGINE.query_archive(
            question=question,
            top_k=top_k,
            filter_by_type=filter_type,
            filter_by_source=filter_source,
            filter_by_speaker=filter_speaker,
            include_raw_chunks=True
        )

        serializable_result = asdict(result)
        
        # Save output logs for the required test question (Deliverable 2)
        if question == "What is the primary definition of success discussed in the files?":
             ATTRIBUTION_ENGINE.save_result_to_json(
                result, 
                "primary_success_definition_output.json"
             )

        return jsonify(serializable_result), 200

    except Exception as e:
        logger.error(f"Query API failed: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to process query",
            "details": str(e),
            "question": question
        }), 500


#--- 6. Main Execution ---

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Berlin Media Archive RAG System')
    parser.add_argument('mode', choices=['serve', 'ingest', 'query_cli'], help='Operation mode')
    parser.add_argument('--audio', type=str, default=CONFIG['audio_file'], help='Path to audio file for ingestion')
    parser.add_argument('--pdf', type=str, default=CONFIG['pdf_file'], help='Path to PDF file for ingestion')
    parser.add_argument('--question', type=str, help='Question for command line query')
    parser.add_argument('--output', type=str, help='Output file for query results (JSON)')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 8080)), help='Port to serve the app on')
    
    args, unknown = parser.parse_known_args()

    if args.mode == 'serve':
        logger.info(f"Starting Flask server on port {args.port}")
        app.run(host='0.0.0.0', port=args.port)

    elif args.mode == 'ingest':
        if not PIPELINE:
            logger.error("Pipeline is not initialized. Check API keys and environment.")
            exit(1)
        PIPELINE.ingest_documents(audio_path=args.audio, pdf_path=args.pdf)

    elif args.mode == 'query_cli':
        if not ATTRIBUTION_ENGINE or not args.question:
            logger.error("Attribution Engine not initialized or --question missing.")
            exit(1)

        print(f"\n{'='*60}")
        print(f"CLI Query: {args.question}")
        print('='*60)

        result = ATTRIBUTION_ENGINE.query_archive(args.question, top_k=5, include_raw_chunks=True)

        print(f"\nAnswer:\n{result.answer}")
        print(f"\nCitations ({len(result.citations)}):")
        for citation in result.citations:
            loc = f"Timestamp: {citation.get('timestamp')}" if citation['type'] == 'audio' else f"Page: {citation.get('page')}"
            print(f"  - [{citation['source']} | {loc}] Score: {citation['relevance_score']:.4f}")

        # Save output logs for the required test question
        if args.question == "What is the primary definition of success discussed in the files?":
            output_path = args.output if args.output else "primary_success_definition_output.json"
            ATTRIBUTION_ENGINE.save_result_to_json(result, output_path)
            print(f"\nRequired output log saved to: {output_path}")
