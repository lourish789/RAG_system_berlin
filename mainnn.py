"""
Berlin Media Archive - Multi-Modal RAG System
Flask Backend with Production Standards
"""
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from services.ingestion import AudioIngestionPipeline, PDFIngestionPipeline
from services.retrieval import HybridRetriever
from services.attribution import AttributionEngine
from services.evaluation import EvaluationService
from utils.error_handler import handle_errors, ArchiveError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('archive.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize services
audio_pipeline = AudioIngestionPipeline()
pdf_pipeline = PDFIngestionPipeline()
retriever = HybridRetriever()
attribution_engine = AttributionEngine(retriever)
evaluator = EvaluationService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Berlin Media Archive',
        'version': '1.0.0'
    })

@app.route('/api/ingest/audio', methods=['POST'])
@handle_errors
def ingest_audio():
    """
    Ingest audio file with transcription and speaker diarization
    
    Expected: multipart/form-data with 'file' field
    """
    if 'file' not in request.files:
        raise ArchiveError('No file provided', 400)
    
    file = request.files['file']
    if file.filename == '':
        raise ArchiveError('Empty filename', 400)
    
    logger.info(f"Starting audio ingestion: {file.filename}")
    
    # Save temporary file
    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)
    
    try:
        # Process audio
        result = audio_pipeline.process(temp_path, file.filename)
        
        logger.info(f"Audio ingestion completed: {result['chunks_created']} chunks created")
        
        return jsonify({
            'success': True,
            'message': 'Audio file processed successfully',
            'details': result
        })
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/ingest/pdf', methods=['POST'])
@handle_errors
def ingest_pdf():
    """
    Ingest PDF document
    
    Expected: multipart/form-data with 'file' field
    """
    if 'file' not in request.files:
        raise ArchiveError('No file provided', 400)
    
    file = request.files['file']
    if file.filename == '':
        raise ArchiveError('Empty filename', 400)
    
    logger.info(f"Starting PDF ingestion: {file.filename}")
    
    # Save temporary file
    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)
    
    try:
        # Process PDF
        result = pdf_pipeline.process(temp_path, file.filename)
        
        logger.info(f"PDF ingestion completed: {result['chunks_created']} chunks created")
        
        return jsonify({
            'success': True,
            'message': 'PDF processed successfully',
            'details': result
        })
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/query', methods=['POST'])
@handle_errors
def query_archive():
    """
    Query the archive with automatic attribution and evaluation
    
    Expected JSON:
    {
        "question": "What did the guest say about technology?",
        "speaker_filter": "guest" (optional),
        "source_filter": "audio" or "pdf" (optional)
    }
    """
    data = request.get_json()
    
    if not data or 'question' not in data:
        raise ArchiveError('Missing question in request', 400)
    
    question = data['question']
    speaker_filter = data.get('speaker_filter')
    source_filter = data.get('source_filter')
    
    logger.info(f"Processing query: {question}")
    
    # Get answer with attribution
    result = attribution_engine.query(
        question=question,
        speaker_filter=speaker_filter,
        source_filter=source_filter
    )
    
    # Evaluate the response
    metrics = evaluator.evaluate_response(
        question=question,
        answer=result['answer'],
        retrieved_contexts=[chunk['text'] for chunk in result['chunks']]
    )
    
    result['metrics'] = metrics
    
    logger.info(f"Query completed - Faithfulness: {metrics['faithfulness']:.2f}, Relevance: {metrics['relevance']:.2f}")
    
    return jsonify(result)

@app.route('/api/evaluate', methods=['POST'])
@handle_errors
def evaluate_batch():
    """
    Batch evaluation endpoint for testing
    
    Expected JSON:
    {
        "test_cases": [
            {
                "question": "...",
                "expected_answer": "..." (optional)
            }
        ]
    }
    """
    data = request.get_json()
    
    if not data or 'test_cases' not in data:
        raise ArchiveError('Missing test_cases in request', 400)
    
    results = []
    
    for test_case in data['test_cases']:
        question = test_case['question']
        
        # Get answer
        result = attribution_engine.query(question=question)
        
        # Evaluate
        metrics = evaluator.evaluate_response(
            question=question,
            answer=result['answer'],
            retrieved_contexts=[chunk['text'] for chunk in result['chunks']]
        )
        
        results.append({
            'question': question,
            'answer': result['answer'],
            'metrics': metrics
        })
    
    avg_faithfulness = sum(r['metrics']['faithfulness'] for r in results) / len(results)
    avg_relevance = sum(r['metrics']['relevance'] for r in results) / len(results)
    
    return jsonify({
        'results': results,
        'summary': {
            'total_cases': len(results),
            'avg_faithfulness': avg_faithfulness,
            'avg_relevance': avg_relevance
        }
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
