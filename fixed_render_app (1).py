import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Flask
from flask import Flask, request, jsonify

# Vector database and AI
from pinecone import Pinecone
import google.generativeai as genai

# Embeddings
from sentence_transformers import SentenceTransformer

# --- Configuration and Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Data Models ---

@dataclass
class QueryResult:
    """Structured query result with citations"""
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    timestamp: str
    query_metadata: Dict[str, Any]

# --- Embedding Service ---

class EmbeddingService:
    """Lightweight embedding service"""
    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        return embedding.tolist()

# --- Attribution Engine ---

class AttributionEngine:
    """Core query engine with citations"""

    def __init__(self, pinecone_api_key: str, gemini_api_key: str, 
                 index_name: str, embedding_model: str):
        logger.info("Initializing Attribution Engine...")

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
            logger.info("Gemini model initialized")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def query_archive(self, question: str, top_k: int = 5,
                     filter_by_type: Optional[str] = None,
                     filter_by_source: Optional[str] = None,
                     filter_by_speaker: Optional[str] = None,
                     include_raw_chunks: bool = True) -> QueryResult:
        """Main query interface"""
        logger.info(f"Query: {question}")
        start_time = datetime.now()

        try:
            # Embed query
            query_embedding = self.embedding_service.embed_single(question)
            
            # Build filters
            filter_dict = self._build_filters(filter_by_type, filter_by_source, filter_by_speaker)
            
            # Retrieve chunks
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            chunks = results.get('matches', [])
            logger.info(f"Retrieved {len(chunks)} chunks")

            if not chunks:
                return self._empty_result(question)

            # Generate answer
            answer_data = self._generate_answer(question, chunks)

            # Build result
            duration = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                question=question,
                answer=answer_data['answer'],
                citations=answer_data['citations'],
                retrieved_chunks=chunks if include_raw_chunks else [],
                timestamp=datetime.now().isoformat(),
                query_metadata={
                    'duration_seconds': duration,
                    'chunks_retrieved': len(chunks),
                    'filters_applied': filter_dict,
                    'top_k': top_k
                }
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return self._empty_result(question, f"Error: {str(e)}")

    def _build_filters(self, filter_by_type, filter_by_source, filter_by_speaker):
        """Build Pinecone metadata filters"""
        filters = {}
        if filter_by_type:
            filters['type'] = {'$eq': filter_by_type}
        if filter_by_source:
            filters['source'] = {'$eq': filter_by_source}
        if filter_by_speaker:
            filters['speaker_id'] = {'$eq': filter_by_speaker}
        return filters if filters else None

    def _generate_answer(self, question: str, chunks: List[Dict]) -> Dict:
        """Generate answer with citations"""
        try:
            context = self._format_context(chunks)
            prompt = f"""You are an archivist assistant. Answer using ONLY the context below.
Cite sources in format: (Source: [filename] at [timestamp/page])

CONTEXT:
{context}

QUESTION: {question}

Provide your answer with proper citations:"""

            response = self.llm.generate_content(prompt)
            citations = self._extract_citations(chunks)

            return {
                'answer': response.text,
                'citations': citations
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'citations': self._extract_citations(chunks)
            }

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format chunks for prompt"""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get('metadata', {})
            text = meta.get('text', 'No content')
            source = meta.get('source', 'unknown')
            chunk_type = meta.get('type', 'unknown')
            
            if chunk_type == 'audio':
                location = f"at {meta.get('timestamp', 'unknown')}"
            else:
                location = f"page {meta.get('page', 'unknown')}"
            
            formatted.append(f"[{i}] {source} ({chunk_type}, {location}): {text}")
        
        return "\n\n".join(formatted)

    def _extract_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Extract citation information"""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get('metadata', {})
            citation = {
                'citation_id': i,
                'source': meta.get('source', 'unknown'),
                'type': meta.get('type', 'unknown'),
                'relevance_score': round(chunk.get('score', 0.0), 4)
            }
            
            if meta.get('type') == 'audio':
                citation['timestamp'] = meta.get('timestamp', 'unknown')
            else:
                citation['page'] = meta.get('page', 'unknown')
            
            text = meta.get('text', '')
            citation['text_snippet'] = (text[:200] + "...") if len(text) > 200 else text
            citations.append(citation)
        
        return citations

    def _empty_result(self, question: str, msg: str = "No relevant information found.") -> QueryResult:
        """Create empty result"""
        return QueryResult(
            question=question,
            answer=msg,
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
        """Save result to JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved to {filepath}")
        except Exception as e:
            logger.error(f"Save failed: {e}")

# --- Flask Application ---

app = Flask(__name__)

# Configuration
CONFIG = {
    'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
    'gemini_api_key': os.getenv('GEMINI_API_KEY'),
    'index_name': os.getenv('PINECONE_INDEX_NAME', 'assess'),
    'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'),
}

ATTRIBUTION_ENGINE = None

def init_services():
    """Initialize services"""
    global ATTRIBUTION_ENGINE
    
    try:
        if not CONFIG['pinecone_api_key']:
            raise ValueError("PINECONE_API_KEY required")
        if not CONFIG['gemini_api_key']:
            raise ValueError("GEMINI_API_KEY required")
        
        logger.info("Initializing services...")
        ATTRIBUTION_ENGINE = AttributionEngine(
            pinecone_api_key=CONFIG['pinecone_api_key'],
            gemini_api_key=CONFIG['gemini_api_key'],
            index_name=CONFIG['index_name'],
            embedding_model=CONFIG['embedding_model']
        )
        logger.info("✓ Services initialized")
        return True
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        return False

# Initialize on startup
init_services()

@app.route('/', methods=['GET'])
def health_check():
    """Health check"""
    if ATTRIBUTION_ENGINE is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Service not initialized"
        }), 500
    
    return jsonify({
        "status": "healthy",
        "service": "Berlin Media Archive RAG",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /"
        }
    }), 200

@app.route('/query', methods=['POST'])
def query_endpoint():
    """Query the archive"""
    if not ATTRIBUTION_ENGINE:
        return jsonify({"error": "Service not initialized"}), 500
    
    try:
        data = request.get_json(silent=True) or {}
        
        if 'question' not in data:
            return jsonify({"error": "Missing 'question' field"}), 400

        result = ATTRIBUTION_ENGINE.query_archive(
            question=data['question'],
            top_k=data.get('top_k', 5),
            filter_by_type=data.get('filter_by_type'),
            filter_by_source=data.get('filter_by_source'),
            filter_by_speaker=data.get('filter_by_speaker'),
            include_raw_chunks=True
        )

        # Save test output if needed
        if data['question'] == "What is the primary definition of success discussed in the files?":
            try:
                ATTRIBUTION_ENGINE.save_result_to_json(result, "test_output.json")
            except:
                pass

        return jsonify(asdict(result)), 200

    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({
            "error": "Query failed",
            "details": str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404"""
    return jsonify({
        "error": "Not found",
        "endpoints": ["GET /", "POST /query"]
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500"""
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

# --- Main ---

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)