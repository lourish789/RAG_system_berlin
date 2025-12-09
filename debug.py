#attribution system 

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

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


@dataclass
class QueryResult:
    """Structured query result with citations"""
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    timestamp: str
    query_metadata: Dict[str, Any]


class AttributionEngine:
    """
    Core query engine that ensures proper attribution and citation
    """

    def __init__(
        self,
        pinecone_api_key: str,
        gemini_api_key: str,
        index_name: str = 'assess',
        embedding_model: str = 'BAAI/bge-large-en-v1.5' # Changed to a 1024-dim model to match Pinecone index
    ):
        """Initialize the attribution engine with all required services"""
        logger.info("Initializing Attribution Engine")

        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {embedding_model} (dimension: {self.embedding_dimension})")

            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")

            # Initialize Gemini
            genai.configure(api_key=gemini_api_key)
            self.llm = genai.GenerativeModel('gemini-pro') # Corrected model name
            logger.info("Initialized Gemini model")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
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
        Main query interface with proper attribution

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

            # Step 2: Build filters
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
            raise

    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            embedding = self.embedding_model.encode(query, convert_to_tensor=False)
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
        filters = {}

        if filter_by_type:
            filters['type'] = {'$eq': filter_by_type}
            logger.info(f"Filtering by type: {filter_by_type}")

        if filter_by_source:
            filters['source'] = {'$eq': filter_by_source}
            logger.info(f"Filtering by source: {filter_by_source}")

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

            matches = results.get('matches', [])

            # Log retrieval details
            if matches:
                logger.info(f"Top match score: {matches[0]['score']:.4f}")
                logger.info(f"Lowest match score: {matches[-1]['score']:.4f}")

            return matches

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Failed to retrieve chunks: {e}")

    def _generate_attributed_answer(
        self,
        question: str,
        chunks: List[Dict]
    ) -> Dict[str, Any]:
        """Generate answer with strict citation requirements"""
        try:
            # Format context
            context = self._format_context_for_prompt(chunks)

            # Create prompt with strict citation rules
            prompt = self._create_attribution_prompt(question, context)

            # Generate response
            response = self.llm.generate_content(prompt)
            answer_text = response.text

            # Extract citations
            citations = self._extract_citations_from_chunks(chunks)

            return {
                'answer': answer_text,
                'citations': citations
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Graceful degradation
            return {
                'answer': f"I encountered an error generating the answer: {str(e)}",
                'citations': []
            }

    def _format_context_for_prompt(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        formatted_chunks = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            text = metadata.get('text', '')
            source = metadata.get('source', 'unknown')
            chunk_type = metadata.get('type', 'unknown')
            score = chunk.get('score', 0.0)

            # Format based on type
            if chunk_type == 'audio':
                timestamp = metadata.get('timestamp', 'unknown')
                location_info = f"at timestamp {timestamp}"
            else:
                page = metadata.get('page', 'unknown')
                location_info = f"on page {page}"

            formatted_chunks.append(
                f"[Chunk {i}] (Relevance: {score:.3f})\n"
                f"Source: {source} ({chunk_type}) {location_info}\n"
                f"Content: {text}\n"
            )

        return "\n---\n".join(formatted_chunks)

    def _create_attribution_prompt(self, question: str, context: str) -> str:
        """Create prompt with strict citation requirements"""
        return f"""You are a professional archivist assistant at the Berlin Media Archive. Your role is to help researchers by providing accurate, well-cited answers from historical documents and audio interviews.\n\nRETRIEVED CONTEXT:\n{context}\n\nRESEARCHER'S QUESTION:\n{question}\n\nCRITICAL INSTRUCTIONS:\n1. Answer ONLY using information from the provided context above\n2. Every factual claim MUST include a citation\n3. Citation format:\n   - For audio: "(Source: [filename] at [timestamp])"\n   - For text: "(Source: [filename], Page [number])"\n4. If multiple sources support a point, cite all of them\n5. If the context doesn't contain enough information to answer the question, explicitly state: "The available archives do not contain sufficient information to answer this question."\n6. Do not make assumptions or add information not present in the context\n7. Be specific and precise in your citations\n\nEXAMPLE OF GOOD CITATION:\n"The architect emphasized the importance of green spaces in urban planning (Source: interview.mp3 at 14:22) and this aligns with the zoning regulations discussed in the document (Source: planning_guide.pdf, Page 4)."\n\nEXAMPLE OF BAD CITATION (DO NOT DO THIS):\n"The speaker discussed urban planning."\n\nNow provide your answer with proper citations:"""

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

            # Add location information
            if metadata.get('type') == 'audio':
                citation['timestamp'] = metadata.get('timestamp', 'unknown')
                citation['start_time'] = metadata.get('start_time', 0)
                citation['end_time'] = metadata.get('end_time', 0)
            else:
                citation['page'] = metadata.get('page', 'unknown')

            # Add snippet
            citation['text_snippet'] = metadata.get('text', '')[:200] + "..."

            citations.append(citation)

        return citations

    def _create_empty_result(self, question: str) -> QueryResult:
        """Create result when no chunks are found"""
        return QueryResult(
            question=question,
            answer="I could not find any relevant information in the archives to answer this question.",
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
            # Manually prepare a serializable dictionary from the QueryResult
            serializable_result = {
                "question": result.question,
                "answer": result.answer,
                "citations": result.citations,
                "timestamp": result.timestamp,
                "query_metadata": result.query_metadata,
                "retrieved_chunks": []
            }

            # Manually extract serializable parts from retrieved_chunks
            for chunk in result.retrieved_chunks:
                serializable_chunk = {
                    "id": chunk.get("id"),
                    "score": chunk.get("score"),
                    "metadata": chunk.get("metadata", {}) # metadata should be a dict and safe
                }
                serializable_result["retrieved_chunks"].append(serializable_chunk)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)

            logger.info(f"Result saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            raise


def query_with_observability(
    engine: AttributionEngine,
    question: str,
    output_file: Optional[str] = None,
    **kwargs
) -> QueryResult:
    """
    Query wrapper with observability features

    Args:
        engine: AttributionEngine instance
        question: Query to process
        output_file: Optional path to save results
        **kwargs: Additional arguments for query_archive

    Returns:
        QueryResult
    """
    logger.info("="*60)
    logger.info(f"NEW QUERY: {question}")
    logger.info("="*60)

    try:
        result = engine.query_archive(question, **kwargs)

        # Log summary
        logger.info(f"Answer generated with {len(result.citations)} citations")
        logger.info(f"Query duration: {result.query_metadata['duration_seconds']:.2f}s")

        # Save if requested
        if output_file:
            engine.save_result_to_json(result, output_file)

        return result

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise


# Example usage
if __name__ == "__main__":
    # Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg")
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw")

    # Initialize engine
    engine = AttributionEngine(
        pinecone_api_key=PINECONE_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        index_name='assess'
    )

    # Example queries
    test_queries = [
        "What is the primary definition of success discussed in the files?",
        "What topics were discussed in the audio interview?",
        "Summarize the main arguments from the text document."
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)

        result = query_with_observability(
            engine,
            query,
            output_file=f"query_result_{i}.json",
            top_k=5
        )

        print(f"\nAnswer:\n{result.answer}")
        print(f"\nCitations ({len(result.citations)}):")
        for citation in result.citations:
            print(f"  - {citation}")
