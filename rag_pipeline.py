"""
rag_pipeline.py - RAG (Retrieval-Augmented Generation) Pipeline
================================================================
This is the "brain" of the chatbot. It does three things:

1. EMBED: Convert text chunks into number arrays (embeddings/vectors)
   so we can compare them mathematically.

2. RETRIEVE: When a user asks a question, find the most similar
   document chunks using cosine similarity.

3. GENERATE: Send the retrieved chunks + question to GPT-4,
   which writes a natural-language answer.

HOW RAG WORKS (simple explanation):
  Without RAG: Ask GPT "What's our vacation policy?"
               GPT guesses based on training data (might be wrong!)

  With RAG:    1. Find the 5 most relevant chunks from your policy docs
               2. Tell GPT: "Here are the actual policies: [chunks]
                             Now answer: What's our vacation policy?"
               3. GPT answers using YOUR actual documents (accurate!)
"""

import logging
import json
import time
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI

from config import settings
from document_processor import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class StoredDocument:
    """
    A document chunk stored in our in-memory "database" with its embedding.

    The embedding is a list of ~1500 numbers that represent the meaning
    of the text. Similar texts have similar embeddings.
    """
    chunk: DocumentChunk           # The text chunk
    embedding: list[float]         # The vector representation

    # Metadata for the API response
    document_name: str = ""
    added_at: float = field(default_factory=time.time)


@dataclass
class ChatResponse:
    """
    The complete response from the chatbot.

    Includes not just the answer, but also which documents were used
    and how confident we are in the answer.
    """
    answer: str                          # The actual answer text
    sources: list[dict]                  # Which documents/chunks were used
    confidence_score: float              # 0.0 to 1.0 (higher = more confident)
    query: str                           # The original question
    model_used: str                      # Which GPT model answered
    tokens_used: int = 0                 # How many OpenAI tokens were consumed
    retrieved_chunks: int = 0            # How many chunks were retrieved


class RAGPipeline:
    """
    The complete RAG system: embed → store → retrieve → generate.

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize()
        pipeline.add_documents(chunks)  # Add your HR policy chunks
        response = pipeline.answer_question("What is the vacation policy?")
        print(response.answer)
    """

    def __init__(self):
        """Set up the pipeline (not connected until initialize() is called)."""
        self.client: Optional[OpenAI] = None
        self.stored_documents: list[StoredDocument] = []
        self.is_ready = False

        # Load settings
        self.chat_model = settings.openai_chat_model
        self.embedding_model = settings.openai_embedding_model
        self.top_k = settings.top_k_results
        self.min_confidence = settings.min_confidence_score
        self.max_context_tokens = settings.max_context_tokens

        logger.info("RAGPipeline created. Call initialize() to connect to OpenAI.")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize(self) -> bool:
        """
        Connect to OpenAI API and verify it works.

        Returns:
            True if ready, False if API key is missing or invalid.
        """
        if not settings.is_openai_configured:
            logger.error(
                "OpenAI API key not configured!\n"
                "Please set OPENAI_API_KEY in your .env file.\n"
                "Get a key at: https://platform.openai.com/api-keys"
            )
            return False

        try:
            self.client = OpenAI(api_key=settings.openai_api_key)

            # Quick test: list models to verify the key works
            self.client.models.list()

            self.is_ready = True
            logger.info(
                f"OpenAI connected successfully!\n"
                f"  Chat model: {self.chat_model}\n"
                f"  Embedding model: {self.embedding_model}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            return False

    # =========================================================================
    # EMBEDDING - Convert text to numbers
    # =========================================================================

    def get_embedding(self, text: str) -> Optional[list[float]]:
        """
        Convert a text string into an embedding vector.

        An embedding is a list of ~1500 numbers that captures the
        "meaning" of the text. Similar meanings → similar numbers.

        Args:
            text: The text to embed (question or document chunk)

        Returns:
            List of floats (the embedding), or None if it fails.
        """
        if not self.is_ready:
            logger.error("Pipeline not initialized. Call initialize() first.")
            return None

        if not text or not text.strip():
            logger.warning("Cannot embed empty text.")
            return None

        # Truncate very long text (embedding models have token limits)
        # 8000 chars ≈ 2000 tokens, well within the 8191 token limit
        text = text[:8000].strip()

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None

    def get_embeddings_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """
        Get embeddings for multiple texts in one API call.

        More efficient than calling get_embedding() in a loop
        because it reduces the number of API round-trips.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embeddings (same order as input).
        """
        if not texts:
            return []

        # Clean and truncate all texts
        clean_texts = [t[:8000].strip() for t in texts if t and t.strip()]

        if not clean_texts:
            return []

        try:
            logger.info(f"Getting embeddings for {len(clean_texts)} texts...")
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=clean_texts,
            )

            # Sort by index to maintain order
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [e.embedding for e in embeddings]

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fall back to individual embeddings
            logger.info("Falling back to individual embeddings...")
            return [self.get_embedding(text) for text in clean_texts]

    # =========================================================================
    # DOCUMENT STORAGE
    # =========================================================================

    def add_documents(self, chunks: list[DocumentChunk]) -> int:
        """
        Embed and store document chunks for later retrieval.

        This is called after syncing documents from Google Drive.
        All chunks are embedded and stored in memory.

        Args:
            chunks: List of DocumentChunk objects from document_processor

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks:
            logger.warning("No chunks provided to add_documents.")
            return 0

        logger.info(f"Adding {len(chunks)} chunks to the pipeline...")

        # Extract texts for batch embedding
        texts = [chunk.text for chunk in chunks]

        # Get all embeddings in one batch call
        embeddings = self.get_embeddings_batch(texts)

        # Store each chunk with its embedding
        added = 0
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is not None:
                stored = StoredDocument(
                    chunk=chunk,
                    embedding=embedding,
                    document_name=chunk.source_file,
                )
                self.stored_documents.append(stored)
                added += 1
            else:
                logger.warning(f"Failed to embed chunk {chunk.chunk_index} from {chunk.source_file}")

        logger.info(
            f"Successfully stored {added}/{len(chunks)} chunks. "
            f"Total documents in memory: {len(self.stored_documents)}"
        )
        return added

    def clear_documents(self):
        """Remove all stored documents (used when re-syncing from scratch)."""
        count = len(self.stored_documents)
        self.stored_documents.clear()
        logger.info(f"Cleared {count} stored documents.")

    # =========================================================================
    # RETRIEVAL - Find relevant chunks
    # =========================================================================

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Returns a score from -1.0 to 1.0:
          1.0 = identical meaning
          0.0 = unrelated
         -1.0 = opposite meaning (rare for text)

        We use numpy for fast vector math.
        """
        a = np.array(vec_a)
        b = np.array(vec_b)

        # Avoid division by zero
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[StoredDocument, float]]:
        """
        Find the document chunks most relevant to a query.

        Process:
          1. Embed the query into a vector
          2. Compare it to every stored chunk's vector
          3. Return the top K most similar chunks

        Args:
            query: The user's question
            top_k: How many chunks to return (defaults to settings.top_k_results)

        Returns:
            List of (StoredDocument, similarity_score) tuples, sorted by score.
        """
        if not self.stored_documents:
            logger.warning("No documents in pipeline. Did you sync documents?")
            return []

        k = top_k or self.top_k

        # Embed the query
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            logger.error("Failed to embed query.")
            return []

        # Calculate similarity to every stored document
        scored_docs = []
        for stored_doc in self.stored_documents:
            score = self._cosine_similarity(query_embedding, stored_doc.embedding)
            scored_docs.append((stored_doc, score))

        # Sort by score (highest first) and take top K
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_docs[:k]

        # Log what we found
        logger.debug(f"Top {k} results for query '{query[:50]}...':")
        for doc, score in top_results:
            logger.debug(f"  [{score:.3f}] {doc.document_name} (chunk {doc.chunk.chunk_index})")

        return top_results

    # =========================================================================
    # GENERATION - Ask GPT to write an answer
    # =========================================================================

    def _build_system_prompt(self) -> str:
        """Build the system prompt that tells GPT how to behave."""
        return """You are a helpful HR Policy Assistant. Your job is to answer
questions about company HR policies accurately and helpfully.

IMPORTANT RULES:
1. ONLY answer based on the HR policy documents provided in the context below.
2. If the answer is not in the provided documents, say so clearly.
   Do NOT make up information or use general knowledge.
3. When citing information, mention which document it came from.
4. Be concise but thorough. Use bullet points for lists.
5. If a policy has specific dates, numbers, or procedures, include them exactly.
6. If you're unsure about something, say "According to the provided documents..."
   to make clear you're citing source material.
7. Be professional, friendly, and helpful."""

    def _build_context(self, retrieved: list[tuple[StoredDocument, float]]) -> str:
        """
        Format retrieved chunks into a context string for GPT.

        Includes source document names so GPT can cite them in answers.
        """
        if not retrieved:
            return "No relevant policy documents found."

        context_parts = ["RELEVANT HR POLICY DOCUMENTS:\n"]

        for i, (stored_doc, score) in enumerate(retrieved, 1):
            context_parts.append(
                f"--- Document {i}: {stored_doc.document_name} "
                f"(Relevance: {score:.0%}) ---\n"
                f"{stored_doc.chunk.text}\n"
            )

        return "\n".join(context_parts)

    def answer_question(self, query: str) -> ChatResponse:
        """
        The main method: takes a question, returns a complete answer.

        Full RAG pipeline:
          1. Retrieve relevant chunks from stored documents
          2. Build a prompt with those chunks as context
          3. Call GPT-4 to generate an answer
          4. Return the answer with metadata

        Args:
            query: The user's question (e.g., "How many vacation days do I get?")

        Returns:
            ChatResponse with answer, sources, and confidence score.
        """
        if not self.is_ready:
            return ChatResponse(
                answer="The chatbot is not initialized. Please check the API key configuration.",
                sources=[],
                confidence_score=0.0,
                query=query,
                model_used="none",
            )

        if not query or not query.strip():
            return ChatResponse(
                answer="Please provide a question.",
                sources=[],
                confidence_score=0.0,
                query=query,
                model_used=self.chat_model,
            )

        logger.info(f"Processing query: {query[:100]}...")

        # Step 1: Retrieve relevant chunks
        retrieved = self.retrieve_relevant_chunks(query)

        if not retrieved:
            return ChatResponse(
                answer=(
                    "I don't have any HR policy documents loaded yet. "
                    "Please sync documents first using the /sync endpoint."
                ),
                sources=[],
                confidence_score=0.0,
                query=query,
                model_used=self.chat_model,
                retrieved_chunks=0,
            )

        # Step 2: Calculate confidence based on top similarity score
        top_score = retrieved[0][1] if retrieved else 0.0

        # Filter out low-confidence results
        confident_results = [
            (doc, score) for doc, score in retrieved
            if score >= self.min_confidence
        ]

        # Step 3: Build context from retrieved chunks
        context = self._build_context(confident_results if confident_results else retrieved[:2])

        # Step 4: Build the user message
        user_message = (
            f"{context}\n\n"
            f"---\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Please answer the question based on the HR policy documents above."
        )

        # Step 5: Call GPT-4
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,   # Low temperature = more factual, less creative
                max_tokens=1000,   # Maximum length of the answer
            )

            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            logger.info(
                f"Generated answer ({tokens_used} tokens used, "
                f"confidence: {top_score:.2f})"
            )

        except Exception as e:
            logger.error(f"GPT call failed: {e}")
            answer = (
                f"I encountered an error while generating the answer: {str(e)}\n"
                "Please check your OpenAI API key and try again."
            )
            tokens_used = 0

        # Step 6: Build sources list for the response
        sources = []
        seen_docs = set()
        for stored_doc, score in retrieved:
            doc_name = stored_doc.document_name
            if doc_name not in seen_docs:
                sources.append({
                    "document": doc_name,
                    "relevance_score": round(score, 3),
                    "chunk_index": stored_doc.chunk.chunk_index,
                    "excerpt": stored_doc.chunk.text[:150] + "...",
                })
                seen_docs.add(doc_name)

        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence_score=round(top_score, 3),
            query=query,
            model_used=self.chat_model,
            tokens_used=tokens_used,
            retrieved_chunks=len(retrieved),
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Return statistics about the current pipeline state."""
        if not self.stored_documents:
            return {
                "status": "empty",
                "total_chunks": 0,
                "total_documents": 0,
                "documents": [],
                "is_ready": self.is_ready,
            }

        # Group chunks by source document
        doc_stats = {}
        for stored in self.stored_documents:
            name = stored.document_name
            if name not in doc_stats:
                doc_stats[name] = {"chunks": 0, "tokens": 0}
            doc_stats[name]["chunks"] += 1
            doc_stats[name]["tokens"] += stored.chunk.token_count

        return {
            "status": "ready",
            "total_chunks": len(self.stored_documents),
            "total_documents": len(doc_stats),
            "documents": [
                {
                    "name": name,
                    "chunks": stats["chunks"],
                    "estimated_tokens": stats["tokens"],
                }
                for name, stats in doc_stats.items()
            ],
            "is_ready": self.is_ready,
            "chat_model": self.chat_model,
            "embedding_model": self.embedding_model,
        }


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pipeline = RAGPipeline()
    if pipeline.initialize():
        print("Pipeline ready!")
        print("Stats:", pipeline.get_stats())
    else:
        print("Failed to initialize. Check OPENAI_API_KEY in .env file.")
