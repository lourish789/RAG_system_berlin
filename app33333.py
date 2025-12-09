import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv

# Third-party libraries
from flask import Flask, request, jsonify
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

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

# --- 2. Data Models ---

@dataclass
class DocumentChunk:
    """A single unit of content for the vector store"""
    id: str
    text: str
    source: str
    type: str  # 'audio' or 'text'
    metadata: Dict[str, Union[str, int, float]]
    embedding: Optional[List[float]] = None

def generate_chunk_id(content_snippet: str, source_file: str, index: int) -> str:
    """Generates a unique ID for a chunk"""
    import hashlib
    content_hash = hashlib.sha256(content_snippet.encode('utf-8')).hexdigest()[:8]
    return f"{source_file.replace('.', '_').replace(' ', '_')}_{content_hash}_{index}"

@dataclass
class QueryResult:
    """Structured query result with citations"""
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    timestamp: str
    query_metadata: Dict[str, Any]

# --- 3. Embedding Service ---

class EmbeddingService:
    """Handles embedding generation"""
    def __init__(self, model_name: str):
        logger.info(f"Loading Embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Single embedding generation failed: {e}")
            raise

# --- 4. Attribution Engine ---

class AttributionEngine:
    """Core query engine with proper attribution and citation"""

    def __init__(
        self,
        pinecone_api_key: str,
        gemini_api_key: str,
        index_name: str = 'assess',
        embedding_model: str = 'BAAI/bge-large-en-v1.5'
    ):
        """Initialize the attribution engine"""
        logger.info("Initializing Attribution Engine")

        try:
            # Initialize embedding model
            self.embedding_service = EmbeddingService(model_name=embedding_model)

            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")

            # Initialize Gemini
            genai.configure(api_key=gemini_api_key)
            self.llm = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Initialized Gemini model")

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise

    def query_archive(
        self,
        question: str,
        top_k: int = 5,
        filter_by_type: Optional[str] = None,
        filter_by_source: Optional[str] = None,
        filter_by_speaker: Optional[str] = None,
        include_raw_chunks: bool = True
    ) -> QueryResult:
        """
        Main query interface with proper attribution

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            filter_by_type: Filter by 'audio' or 'text'
            filter_by_source: Filter by specific source file
            filter_by_speaker: Filter by speaker ID
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

            # Step 2: Build filters
            filter_dict = self._build_filters(filter_by_type, filter_by_source, filter_by_speaker)

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

            # Step 5: Create structured result
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
            return self._create_empty_result(
                question, 
                f"A system error occurred: {str(e)}"
            )

    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            return self.embedding_service.embed_single(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise RuntimeError(f"Failed to embed query: {e}")

    def _build_filters(
        self,
        filter_by_type: Optional[str],
        filter_by_source: Optional[str],
        filter_by_speaker: Optional[str]
    ) -> Optional[Dict]:
        """Build Pinecone metadata filters"""
        filters = {}

        if filter_by_type:
            filters['type'] = {'$eq': filter_by_type}

        if filter_by_source:
            filters['source'] = {'$eq': filter_by_source}
            
        if filter_by_speaker:
            filters['speaker_id'] = {'$eq': filter_by_speaker}
            
        return filters if filters else None

    def _retrieve_chunks(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_dict: Optional[Dict]
    ) -> List[Dict]:
        """Retrieve chunks from vector store"""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            matches = results.get('matches', [])
            logger.info(f"Pinecone returned {len(matches)} matches")
            
            return matches

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
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

            logger.info("Generating answer with Gemini...")
            response = self.llm.generate_content(prompt)
            answer_text = response.text

            citations = self._extract_citations_from_chunks(chunks)

            return {
                'answer': answer_text,
                'citations': citations
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            return {
                'answer': f"I encountered an error generating the answer: {str(e)}",
                'citations': self._extract_citations_from_chunks(chunks)
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
            else:  # type == 'text'
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
        return f"""You are a professional archivist assistant at the Berlin Media Archive. Your role is to help researchers by providing accurate, well-cited answers from historical documents and audio interviews.

RETRIEVED CONTEXT:
{context}

RESEARCHER'S QUESTION:
{question}

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context above
2. Every factual claim MUST include a citation
3. Citation format:
   - For audio: "(Source: [filename] at [timestamp])"
   - For text: "(Source: [filename], Page [number])"
4. If multiple sources support a point, cite all of them
5. If the context doesn't contain enough information to answer the question, explicitly state: "The available archives do not contain sufficient information to answer this question."
6. Do not make assumptions or add information not present in the context

Now provide your answer with proper citations:"""

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
            
            text_content = metadata.get('text', '')
            citation['text_snippet'] = (text_content[:200] + "...") if len(text_content) > 200 else text_content
            citations.append(citation)
        return citations

    def _create_empty_result(
        self, 
        question: str, 
        error_message: str = "I could not find any relevant information in the archives to answer this question."
    ) -> QueryResult:
        """Create result when no chunks are found or on error"""
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

    def save_result_to_json(self, result: QueryResult, filepath: str):
        """Save query result to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")

# --- 5. Flask Application ---

# Get config from environment variables
CONFIG = {
    'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
    'gemini_api_key': os.getenv('GEMINI_API_KEY'),
    'index_name': os.getenv('PINECONE_INDEX_NAME', 'assess'),
    'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5'),
}

app = Flask(__name__)

# Initialize Attribution Engine
ATTRIBUTION_ENGINE = None

def initialize_services():
    """Initialize services with proper error handling"""
    global ATTRIBUTION_ENGINE
    
    try:
        # Validate required environment variables
        if not CONFIG['pinecone_api_key']:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not CONFIG['gemini_api_key']:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        logger.info("Initializing Attribution Engine...")
        ATTRIBUTION_ENGINE = AttributionEngine(
            pinecone_api_key=CONFIG['pinecone_api_key'],
            gemini_api_key=CONFIG['gemini_api_key'],
            index_name=CONFIG['index_name'],
            embedding_model=CONFIG['embedding_model']
        )
        logger.info("Attribution Engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"FATAL: Could not initialize services: {e}", exc_info=True)
        return False

# Initialize on startup
initialize_services()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if ATTRIBUTION_ENGINE is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Attribution Engine not initialized. Check environment variables."
        }), 500
    
    return jsonify({
        "status": "healthy",
        "message": "Berlin Media Archive RAG System is running",
        "endpoints": {
            "query": "/query (POST)",
            "health": "/ (GET)"
        }
    }), 200

@app.route('/query', methods=['POST'])
def query_archive_api():
    """Endpoint for querying the archive"""
    if not ATTRIBUTION_ENGINE:
        return jsonify({
            "error": "Attribution Engine not initialized. Check server logs and environment variables."
        }), 500
    
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        
        if 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        question = data['question']
        top_k = data.get('top_k', 5)
        filter_type = data.get('filter_by_type')
        filter_source = data.get('filter_by_source')
        filter_speaker = data.get('filter_by_speaker')

        logger.info(f"Received query: {question}")

        result = ATTRIBUTION_ENGINE.query_archive(
            question=question,
            top_k=top_k,
            filter_by_type=filter_type,
            filter_by_source=filter_source,
            filter_by_speaker=filter_speaker,
            include_raw_chunks=True
        )

        serializable_result = asdict(result)
        
        # Save output for specific test question
        if question == "What is the primary definition of success discussed in the files?":
            try:
                ATTRIBUTION_ENGINE.save_result_to_json(
                    result, 
                    "primary_success_definition_output.json"
                )
            except Exception as e:
                logger.warning(f"Could not save test output: {e}")

        return jsonify(serializable_result), 200

    except Exception as e:
        logger.error(f"Query API failed: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to process query",
            "details": str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": {
            "health": "GET /",
            "query": "POST /query"
        }
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "Check server logs for details"
    }), 500

# --- 6. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Berlin Media Archive RAG System')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 8080)), help='Port to serve the app on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args, unknown = parser.parse_known_args()

    logger.info(f"Starting Flask server on port {args.port}")
    app.run(
        host='0.0.0.0', 
        port=args.port, 
        debug=args.debug
    )
