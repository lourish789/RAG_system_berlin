import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import sys
import traceback

# Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

# Vector database and AI
from pinecone import Pinecone
import google.generativeai as genai

# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Configuration and Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('archive_rag.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Data Models ---

@dataclass
class Citation:
    """Individual citation with full metadata"""
    citation_id: int
    source: str
    content_type: str  # 'audio' or 'text'
    location: str  # timestamp for audio, page for text
    text_snippet: str
    relevance_score: float
    speaker_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class QueryResult:
    """Complete query result with strict attribution"""
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    timestamp: str
    query_metadata: Dict[str, Any]
    retrieval_trace: Dict[str, Any]  # For observability

# --- Embedding Service ---

class EmbeddingService:
    """Google-based embedding service (no model download required)"""
    
    def __init__(self, google_api_key: str):
        logger.info("Initializing Google Embeddings API...")
        try:
            self.model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
            )
            # Test embedding
            test_embed = self.model.embed_query("test")
            logger.info(f"✓ Embeddings initialized. Dimension: {len(test_embed)}")
        except Exception as e:
            logger.error(f"✗ Embedding initialization failed: {e}")
            raise

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text with error handling"""
        try:
            if not text or not text.strip():
                raise ValueError("Empty text provided for embedding")
            
            embedding = self.model.embed_query(text[:8000])  # Limit length
            return embedding
        except Exception as e:
            logger.error(f"Embedding error for text: {text[:50]}... - {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding for efficiency"""
        try:
            embeddings = self.model.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            raise

# --- Attribution Engine ---

class AttributionEngine:
    """
    Production-grade RAG system with strict attribution requirements.
    Implements all core requirements from the Berlin Archive assessment.
    """

    def __init__(self, pinecone_api_key: str, gemini_api_key: str, index_name: str):
        logger.info("=" * 70)
        logger.info("BERLIN MEDIA ARCHIVE - ATTRIBUTION ENGINE")
        logger.info("=" * 70)
        
        try:
            # Step 1: Initialize embeddings
            logger.info("Step 1/3: Initializing embedding service...")
            self.embedding_service = EmbeddingService(google_api_key=gemini_api_key)
            logger.info("✓ Embeddings ready")

            # Step 2: Connect to Pinecone
            logger.info("Step 2/3: Connecting to vector database...")
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(index_name)
            
            # Verify connection
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            logger.info(f"✓ Pinecone connected: {index_name}")
            logger.info(f"  - Total vectors: {total_vectors}")
            logger.info(f"  - Dimension: {stats.get('dimension', 'unknown')}")

            # Step 3: Initialize LLM
            logger.info("Step 3/3: Initializing language model...")
            genai.configure(api_key=gemini_api_key)
            self.llm = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    'temperature': 0.1,  # Low temperature for factual responses
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
            logger.info("✓ Gemini LLM initialized")

            logger.info("=" * 70)
            logger.info("✓ ATTRIBUTION ENGINE READY")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"✗ Initialization failed: {e}")
            logger.error(traceback.format_exc())
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
        Main query interface with strict attribution requirements.
        
        Args:
            question: User's research question
            top_k: Number of chunks to retrieve
            filter_by_type: Filter by 'audio' or 'text'
            filter_by_source: Filter by specific source file
            filter_by_speaker: Filter by speaker ID (for audio)
            include_raw_chunks: Include raw retrieved chunks in response
            
        Returns:
            QueryResult with answer and strict citations
        """
        logger.info("=" * 70)
        logger.info(f"NEW QUERY: {question}")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        retrieval_trace = {
            'query': question,
            'start_time': start_time.isoformat(),
            'filters_applied': {},
            'retrieval_steps': []
        }

        try:
            # Step 1: Embed the query
            logger.info("Step 1: Embedding query...")
            try:
                query_embedding = self.embedding_service.embed_single(question)
                retrieval_trace['retrieval_steps'].append({
                    'step': 'embedding',
                    'status': 'success',
                    'embedding_dim': len(query_embedding)
                })
                logger.info(f"✓ Query embedded (dim: {len(query_embedding)})")
            except Exception as e:
                logger.error(f"✗ Embedding failed: {e}")
                return self._error_result(question, f"Embedding error: {str(e)}", retrieval_trace)

            # Step 2: Build metadata filters
            logger.info("Step 2: Building filters...")
            filter_dict = self._build_filters(filter_by_type, filter_by_source, filter_by_speaker)
            retrieval_trace['filters_applied'] = filter_dict or {}
            
            if filter_dict:
                logger.info(f"  - Filters: {filter_dict}")
            else:
                logger.info("  - No filters applied")

            # Step 3: Query vector database
            logger.info(f"Step 3: Querying vector store (top_k={top_k})...")
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
                
                chunks = results.get('matches', [])
                logger.info(f"✓ Retrieved {len(chunks)} chunks")
                
                retrieval_trace['retrieval_steps'].append({
                    'step': 'vector_search',
                    'status': 'success',
                    'chunks_retrieved': len(chunks),
                    'top_scores': [round(c.get('score', 0), 4) for c in chunks[:3]]
                })
                
                # Log retrieved chunks for observability
                for i, chunk in enumerate(chunks, 1):
                    meta = chunk.get('metadata', {})
                    score = chunk.get('score', 0)
                    logger.info(f"  [{i}] {meta.get('source', 'unknown')} "
                              f"({meta.get('type', 'unknown')}) - Score: {score:.4f}")

            except Exception as e:
                logger.error(f"✗ Vector search failed: {e}")
                return self._error_result(question, f"Search error: {str(e)}", retrieval_trace)

            # Handle empty results
            if not chunks:
                logger.warning("⚠ No relevant chunks found")
                return self._empty_result(question, retrieval_trace)

            # Step 4: Generate answer with strict attribution
            logger.info("Step 4: Generating attributed answer...")
            try:
                answer_data = self._generate_attributed_answer(question, chunks)
                retrieval_trace['retrieval_steps'].append({
                    'step': 'answer_generation',
                    'status': 'success',
                    'citations_generated': len(answer_data['citations'])
                })
                logger.info(f"✓ Answer generated with {len(answer_data['citations'])} citations")
            except Exception as e:
                logger.error(f"✗ Answer generation failed: {e}")
                return self._error_result(
                    question, 
                    f"Answer generation error: {str(e)}", 
                    retrieval_trace,
                    chunks=chunks
                )

            # Step 5: Build final result
            duration = (datetime.now() - start_time).total_seconds()
            retrieval_trace['end_time'] = datetime.now().isoformat()
            retrieval_trace['duration_seconds'] = duration

            logger.info("=" * 70)
            logger.info(f"✓ QUERY COMPLETE ({duration:.2f}s)")
            logger.info("=" * 70)

            return QueryResult(
                question=question,
                answer=answer_data['answer'],
                citations=answer_data['citations'],
                retrieved_chunks=chunks if include_raw_chunks else [],
                timestamp=datetime.now().isoformat(),
                query_metadata={
                    'duration_seconds': round(duration, 3),
                    'chunks_retrieved': len(chunks),
                    'citations_count': len(answer_data['citations']),
                    'filters_applied': filter_dict,
                    'top_k': top_k,
                    'average_relevance_score': round(
                        sum(c.get('score', 0) for c in chunks) / len(chunks) if chunks else 0,
                        4
                    )
                },
                retrieval_trace=retrieval_trace
            )

        except Exception as e:
            logger.error(f"✗ Query failed with unexpected error: {e}")
            logger.error(traceback.format_exc())
            return self._error_result(question, f"Unexpected error: {str(e)}", retrieval_trace)

    def _build_filters(
        self,
        filter_by_type: Optional[str],
        filter_by_source: Optional[str],
        filter_by_speaker: Optional[str]
    ) -> Optional[Dict]:
        """
        Build Pinecone metadata filters.
        Implements Module A (Search Engineering) - Metadata Filtering.
        """
        filters = {}
        
        if filter_by_type:
            if filter_by_type not in ['audio', 'text']:
                logger.warning(f"Invalid type filter: {filter_by_type}")
            else:
                filters['type'] = {'$eq': filter_by_type}
        
        if filter_by_source:
            filters['source'] = {'$eq': filter_by_source}
        
        if filter_by_speaker:
            # Module B: Speaker Diarization support
            filters['speaker_id'] = {'$eq': filter_by_speaker}
        
        return filters if filters else None

    def _generate_attributed_answer(self, question: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer with STRICT attribution requirements.
        Implements Part 1, Requirement 3: The Attribution Engine.
        
        Requirements:
        - Answer must use ONLY retrieved context
        - Must cite sources with specific locations
        - Format: (Source: filename at timestamp/page)
        """
        try:
            # Format context with explicit citation markers
            context = self._format_context_with_citations(chunks)
            
            # Create strict prompt enforcing attribution
            prompt = f"""You are a research assistant for the Berlin Media Archive. Your task is to answer questions using ONLY the provided context.

STRICT RULES:
1. Use ONLY information from the context below
2. ALWAYS cite your sources using this format:
   - For audio: (Source: filename at MM:SS)
   - For text: (Source: filename, Page X)
3. If multiple sources support a claim, cite all of them
4. If the context doesn't contain the answer, say "The archive does not contain information about this topic."
5. Never make assumptions or add information not in the context

CONTEXT WITH CITATIONS:
{context}

RESEARCH QUESTION: {question}

Provide a detailed answer with proper citations:"""

            # Generate response
            response = self.llm.generate_content(prompt)
            answer_text = response.text.strip()
            
            # Extract structured citations
            citations = self._extract_structured_citations(chunks)
            
            # Validate that answer contains citations
            if not self._validate_citations_in_answer(answer_text):
                logger.warning("⚠ Generated answer lacks proper citations")
            
            return {
                'answer': answer_text,
                'citations': citations
            }
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            # Graceful degradation: return citations even if LLM fails
            return {
                'answer': f"Error generating answer: {str(e)}. Please review the citations below for relevant information.",
                'citations': self._extract_structured_citations(chunks)
            }

    def _format_context_with_citations(self, chunks: List[Dict]) -> str:
        """Format chunks with explicit citation markers for the LLM"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get('metadata', {})
            text = meta.get('text', 'No content available')
            source = meta.get('source', 'unknown')
            content_type = meta.get('type', 'unknown')
            
            # Format location based on type
            if content_type == 'audio':
                timestamp = meta.get('timestamp', 'unknown')
                location = f"at {timestamp}"
                speaker = meta.get('speaker_id', '')
                speaker_info = f" (Speaker: {speaker})" if speaker else ""
            else:
                page = meta.get('page', 'unknown')
                location = f"Page {page}"
                speaker_info = ""
            
            # Create citation marker
            citation = f"[{i}] Source: {source} {location}{speaker_info}"
            
            formatted_chunks.append(f"{citation}\nContent: {text}\n")
        
        return "\n".join(formatted_chunks)

    def _extract_structured_citations(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract structured citation information.
        Returns list of dicts with all required metadata.
        """
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get('metadata', {})
            content_type = meta.get('type', 'unknown')
            
            citation = {
                'citation_id': i,
                'source': meta.get('source', 'unknown'),
                'type': content_type,
                'relevance_score': round(chunk.get('score', 0.0), 4)
            }
            
            # Add type-specific location
            if content_type == 'audio':
                citation['timestamp'] = meta.get('timestamp', 'unknown')
                citation['speaker_id'] = meta.get('speaker_id', None)
            else:
                citation['page'] = meta.get('page', 'unknown')
            
            # Add text snippet
            text = meta.get('text', '')
            citation['text_snippet'] = (text[:250] + "...") if len(text) > 250 else text
            
            # Add full metadata for observability
            citation['metadata'] = meta
            
            citations.append(citation)
        
        return citations

    def _validate_citations_in_answer(self, answer: str) -> bool:
        """Check if answer contains proper citations"""
        # Basic validation: check for common citation patterns
        has_source_marker = 'Source:' in answer or 'source:' in answer
        has_page_or_time = 'Page' in answer or ':' in answer  # Check for page or timestamp
        return has_source_marker and has_page_or_time

    def _empty_result(self, question: str, retrieval_trace: Dict) -> QueryResult:
        """Return when no relevant chunks are found"""
        return QueryResult(
            question=question,
            answer="No relevant information found in the archive. Please try rephrasing your question or using different search terms.",
            citations=[],
            retrieved_chunks=[],
            timestamp=datetime.now().isoformat(),
            query_metadata={
                'duration_seconds': 0,
                'chunks_retrieved': 0,
                'citations_count': 0,
                'filters_applied': None,
                'top_k': 0
            },
            retrieval_trace=retrieval_trace
        )

    def _error_result(
        self,
        question: str,
        error_message: str,
        retrieval_trace: Dict,
        chunks: List[Dict] = None
    ) -> QueryResult:
        """Return error result with graceful degradation"""
        citations = self._extract_structured_citations(chunks) if chunks else []
        
        return QueryResult(
            question=question,
            answer=f"Error: {error_message}",
            citations=citations,
            retrieved_chunks=chunks or [],
            timestamp=datetime.now().isoformat(),
            query_metadata={
                'error': error_message,
                'chunks_retrieved': len(chunks) if chunks else 0,
                'citations_count': len(citations)
            },
            retrieval_trace=retrieval_trace
        )

    def save_result_to_json(self, result: QueryResult, filepath: str):
        """Save query result to JSON file for logging/testing"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Result saved to {filepath}")
        except Exception as e:
            logger.error(f"✗ Failed to save result: {e}")

# --- Flask Application ---

app = Flask(__name__)

# Enable CORS for all origins (adjust for production)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration from environment variables
CONFIG = {
    'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
    'gemini_api_key': os.getenv('GEMINI_API_KEY'),
    'index_name': os.getenv('PINECONE_INDEX_NAME', 'assess'),
}

# Global state
ATTRIBUTION_ENGINE = None
INITIALIZATION_ERROR = None
IS_INITIALIZING = True

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global IS_INITIALIZING, INITIALIZATION_ERROR, ATTRIBUTION_ENGINE
    
    if INITIALIZATION_ERROR:
        return jsonify({
            "status": "unhealthy",
            "message": "Initialization failed",
            "error": str(INITIALIZATION_ERROR)
        }), 500
    
    if IS_INITIALIZING:
        return jsonify({
            "status": "initializing",
            "message": "Attribution engine is starting up...",
            "service": "Berlin Media Archive RAG System"
        }), 200
    
    if ATTRIBUTION_ENGINE is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Service not initialized"
        }), 500
    
    return jsonify({
        "status": "healthy",
        "service": "Berlin Media Archive RAG System",
        "version": "1.0.0",
        "features": [
            "Multi-modal search (audio + text)",
            "Strict attribution with citations",
            "Speaker filtering (diarization support)",
            "Metadata filtering",
            "Observability tracing"
        ],
        "endpoints": {
            "query": "POST /query",
            "health": "GET /",
            "status": "GET /status"
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Simple health endpoint for Render"""
    global ATTRIBUTION_ENGINE, IS_INITIALIZING
    
    if IS_INITIALIZING:
        return jsonify({"status": "initializing"}), 200
    
    if ATTRIBUTION_ENGINE:
        return jsonify({"status": "healthy"}), 200
    
    return jsonify({"status": "unhealthy"}), 500


@app.route('/status', methods=['GET'])
def status_check():
    """Detailed status endpoint"""
    global IS_INITIALIZING, INITIALIZATION_ERROR, ATTRIBUTION_ENGINE
    
    return jsonify({
        "is_initializing": IS_INITIALIZING,
        "engine_ready": ATTRIBUTION_ENGINE is not None,
        "has_error": INITIALIZATION_ERROR is not None,
        "error": str(INITIALIZATION_ERROR) if INITIALIZATION_ERROR else None,
        "config": {
            "index_name": CONFIG.get('index_name'),
            "has_pinecone_key": bool(CONFIG.get('pinecone_api_key')),
            "has_gemini_key": bool(CONFIG.get('gemini_api_key'))
        },
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/query', methods=['POST', 'OPTIONS'])
def query_endpoint():
    """
    Main query endpoint for the Berlin Archive.
    
    Request body:
    {
        "question": str (required),
        "top_k": int (optional, default=5),
        "filter_by_type": str (optional, 'audio' or 'text'),
        "filter_by_source": str (optional, filename),
        "filter_by_speaker": str (optional, speaker ID),
        "include_raw_chunks": bool (optional, default=false)
    }
    """
    global IS_INITIALIZING, ATTRIBUTION_ENGINE
    
    # Handle preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    # Check initialization
    if IS_INITIALIZING:
        return jsonify({
            "error": "Service is still initializing",
            "message": "Please wait a moment and try again"
        }), 503
    
    if not ATTRIBUTION_ENGINE:
        return jsonify({
            "error": "Service not initialized",
            "message": "Attribution engine failed to start"
        }), 500
    
    try:
        # Parse request
        data = request.get_json(silent=True) or {}
        
        if 'question' not in data:
            return jsonify({
                "error": "Missing required field: 'question'"
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                "error": "Question cannot be empty"
            }), 400
        
        logger.info(f"API Request: {question[:100]}...")
        
        # Query the archive
        result = ATTRIBUTION_ENGINE.query_archive(
            question=question,
            top_k=data.get('top_k', 5),
            filter_by_type=data.get('filter_by_type'),
            filter_by_source=data.get('filter_by_source'),
            filter_by_speaker=data.get('filter_by_speaker'),
            include_raw_chunks=data.get('include_raw_chunks', False)
        )
        
        # Special case: save test query for assessment
        if "primary definition of success" in question.lower():
            try:
                ATTRIBUTION_ENGINE.save_result_to_json(result, "test_output.json")
                logger.info("✓ Test output saved to test_output.json")
            except Exception as e:
                logger.warning(f"Could not save test output: {e}")
        
        return jsonify(asdict(result)), 200
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Query processing failed",
            "details": str(e)
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["GET /", "GET /health", "GET /status", "POST /query"]
    }), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({
        "error": "Internal server error"
    }), 500


def init_services():
    """Initialize attribution engine with timeout protection"""
    global ATTRIBUTION_ENGINE, INITIALIZATION_ERROR, IS_INITIALIZING
    
    try:
        logger.info("=" * 70)
        logger.info("BERLIN MEDIA ARCHIVE - INITIALIZATION")
        logger.info("=" * 70)
        
        # Validate configuration
        if not CONFIG['pinecone_api_key']:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not CONFIG['gemini_api_key']:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        logger.info("✓ Environment variables validated")
        logger.info(f"✓ Index name: {CONFIG['index_name']}")
        
        # Initialize attribution engine
        ATTRIBUTION_ENGINE = AttributionEngine(
            pinecone_api_key=CONFIG['pinecone_api_key'],
            gemini_api_key=CONFIG['gemini_api_key'],
            index_name=CONFIG['index_name']
        )
        
        IS_INITIALIZING = False
        logger.info("=" * 70)
        logger.info("✓ SYSTEM READY FOR QUERIES")
        logger.info("=" * 70)
        return True
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"✗ INITIALIZATION FAILED: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 70)
        INITIALIZATION_ERROR = e
        IS_INITIALIZING = False
        return False


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    
    logger.info("=" * 70)
    logger.info("BERLIN MEDIA ARCHIVE - RAG SYSTEM")
    logger.info(f"Starting server on port {port}")
    logger.info("=" * 70)
    
    # Start initialization in background
    import threading
    init_thread = threading.Thread(target=init_services, daemon=False)
    init_thread.start()
    
    logger.info("✓ Initialization thread started")
    logger.info("✓ Server starting with CORS enabled")
    logger.info("=" * 70)
    
    # Start Flask application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )
