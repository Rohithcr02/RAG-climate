import os
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import numpy as np

load_dotenv()


class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize local ChromaDB client
        chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Get collection
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "hvac_documents")
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Connected to collection: {self.collection_name}")
        except Exception as e:
            raise Exception(f"Could not connect to collection '{self.collection_name}': {e}")

        # Cache for BM25 (will be loaded on first search)
        self.bm25 = None
        self.all_documents = None
        self.all_metadatas = None
        self.all_ids = None

    def _load_bm25_index(self, brand_filter: str = None):
        """
        Load all documents and create BM25 index.

        Args:
            brand_filter: Optional brand name to filter documents
        """
        print("Loading documents for BM25 indexing...")

        # Get all documents from ChromaDB
        results = self.collection.get(
            include=['documents', 'metadatas']
        )

        documents = results['documents']
        metadatas = results['metadatas']
        ids = results['ids']

        # Apply brand filter if specified
        if brand_filter:
            filtered_data = [
                (doc, meta, doc_id)
                for doc, meta, doc_id in zip(documents, metadatas, ids)
                if brand_filter.lower() in meta.get('filename', '').lower()
            ]
            if filtered_data:
                documents, metadatas, ids = zip(*filtered_data)
                documents = list(documents)
                metadatas = list(metadatas)
                ids = list(ids)
            else:
                print(f"Warning: No documents found for brand '{brand_filter}'")
                documents, metadatas, ids = [], [], []

        self.all_documents = documents
        self.all_metadatas = metadatas
        self.all_ids = ids

        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"BM25 index created with {len(documents)} documents")

    def vector_search(self, query: str, top_k: int = 20, brand_filter: str = None) -> List[Dict]:
        """
        Perform vector similarity search using ChromaDB.

        Args:
            query: Search query
            top_k: Number of results to return
            brand_filter: Optional brand name to filter documents

        Returns:
            List of results with documents and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Prepare where filter for brand
        where_filter = None
        if brand_filter:
            where_filter = {
                "filename": {
                    "$contains": brand_filter.lower()
                }
            }

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'method': 'vector'
            })

        return search_results

    def bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with documents and metadata
        """
        if not self.bm25 or not self.all_documents:
            raise Exception("BM25 index not loaded. This should not happen.")

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Format results
        search_results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                search_results.append({
                    'id': self.all_ids[idx],
                    'document': self.all_documents[idx],
                    'metadata': self.all_metadatas[idx],
                    'score': float(scores[idx]),
                    'method': 'bm25'
                })

        return search_results

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF formula: RRF(d) = sum(1 / (k + rank(d)))
        where k is a constant (typically 60) and rank(d) is the rank of document d.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: RRF constant (default 60)

        Returns:
            Merged and ranked results
        """
        # Calculate RRF scores
        rrf_scores = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'score': 0,
                    'document': result['document'],
                    'metadata': result['metadata']
                }
            rrf_scores[doc_id]['score'] += 1 / (k + rank)

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'score': 0,
                    'document': result['document'],
                    'metadata': result['metadata']
                }
            rrf_scores[doc_id]['score'] += 1 / (k + rank)

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Format results
        merged_results = []
        for doc_id, data in sorted_results:
            merged_results.append({
                'id': doc_id,
                'document': data['document'],
                'metadata': data['metadata'],
                'rrf_score': data['score']
            })

        return merged_results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        brand_filter: str = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector and BM25 search with RRF.

        Args:
            query: Search query
            top_k: Number of final results to return
            brand_filter: Optional brand name to filter documents

        Returns:
            Top-k merged results
        """
        print(f"\nSearching for: '{query}'")
        if brand_filter:
            print(f"Filtering by brand: '{brand_filter}'")

        # Load BM25 index (will use cached version if already loaded with same filter)
        if self.bm25 is None or brand_filter:
            self._load_bm25_index(brand_filter)

        if not self.all_documents:
            print("No documents available for search")
            return []

        # Perform vector search
        print("Running vector search...")
        vector_results = self.vector_search(query, top_k=20, brand_filter=brand_filter)

        # Perform BM25 search
        print("Running BM25 search...")
        bm25_results = self.bm25_search(query, top_k=20)

        # Merge with RRF
        print("Merging results with Reciprocal Rank Fusion...")
        merged_results = self.reciprocal_rank_fusion(vector_results, bm25_results)

        # Return top-k
        final_results = merged_results[:top_k]
        print(f"Returning top {len(final_results)} results")

        return final_results

    def get_available_brands(self) -> List[str]:
        """
        Get list of unique brands (filenames) in the collection.

        Returns:
            List of unique filenames
        """
        results = self.collection.get(include=['metadatas'])
        filenames = set(meta['filename'] for meta in results['metadatas'])

        # Extract brand names (assuming format: BrandName_Model.pdf)
        brands = set()
        for filename in filenames:
            # Remove .pdf extension and extract brand (part before underscore or space)
            name = filename.replace('.pdf', '')
            brand = name.split('_')[0].split()[0]
            brands.add(brand)

        return sorted(list(brands))


def main():
    """Test the retrieval system."""
    retriever = HybridRetriever()

    # Get available brands
    print("\nAvailable brands:")
    brands = retriever.get_available_brands()
    for brand in brands:
        print(f"  - {brand}")

    # Test query
    test_query = "How do I troubleshoot a refrigerant leak?"
    results = retriever.hybrid_search(test_query, top_k=5)

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['rrf_score']:.4f}")
        print(f"Document: {result['metadata']['filename']}")
        print(f"Page: {result['metadata']['page_number']}")
        print(f"Chunk: {result['metadata']['chunk_index']}")
        print(f"Preview: {result['document'][:200]}...")


if __name__ == "__main__":
    main()
