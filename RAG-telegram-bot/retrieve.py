import os
import time as _time
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

        chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "hvac_documents")
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Connected to collection: {self.collection_name}")
        except Exception as e:
            raise Exception(f"Could not connect to collection '{self.collection_name}': {e}")

        self.bm25 = None
        self.all_documents = None
        self.all_metadatas = None
        self.all_ids = None

    def _load_bm25_index(self, brand_filter: str = None):
        """Load docs from ChromaDB and build BM25 index."""
        print("Loading documents for BM25 indexing...")


        results = self.collection.get(
            include=['documents', 'metadatas']
        )

        documents = results['documents']
        metadatas = results['metadatas']
        ids = results['ids']


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


        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"BM25 index created with {len(documents)} documents")

    def vector_search(self, query: str, top_k: int = 20, brand_filter: str = None) -> List[Dict]:
        """Semantic search via ChromaDB embeddings."""
        query_embedding = self.embedding_model.encode(query).tolist()

        where_filter = None
        if brand_filter:
            where_filter = {
                "filename": {
                    "$contains": brand_filter.lower()
                }
            }


        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )


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
        """Keyword search via BM25."""
        if not self.bm25 or not self.all_documents:
            raise Exception("BM25 index not loaded. This should not happen.")


        tokenized_query = query.lower().split()


        scores = self.bm25.get_scores(tokenized_query)


        top_indices = np.argsort(scores)[::-1][:top_k]


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
        """Merge vector + BM25 results using RRF. score = sum(1 / (k + rank))."""

        rrf_scores = {}


        for rank, result in enumerate(vector_results, start=1):
            doc_id = result['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'score': 0,
                    'document': result['document'],
                    'metadata': result['metadata']
                }
            rrf_scores[doc_id]['score'] += 1 / (k + rank)


        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'score': 0,
                    'document': result['document'],
                    'metadata': result['metadata']
                }
            rrf_scores[doc_id]['score'] += 1 / (k + rank)


        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )


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
        """Run vector + BM25 search, merge with RRF, return top_k results."""
        print(f"\nSearching for: '{query}'")
        if brand_filter:
            print(f"Filtering by brand: '{brand_filter}'")


        if self.bm25 is None or brand_filter:
            self._load_bm25_index(brand_filter)

        if not self.all_documents:
            print("No documents available for search")
            return []


        print("Running vector search...")
        vector_results = self.vector_search(query, top_k=20, brand_filter=brand_filter)


        print("Running BM25 search...")
        bm25_results = self.bm25_search(query, top_k=20)


        print("Merging results with Reciprocal Rank Fusion...")
        merged_results = self.reciprocal_rank_fusion(vector_results, bm25_results)


        final_results = merged_results[:top_k]
        print(f"Returning top {len(final_results)} results")

        return final_results

    def hybrid_search_timed(
        self,
        query: str,
        top_k: int = 5,
        brand_filter: str = None,
    ) -> Dict:
        """
        Same as hybrid_search but returns timing data.
        Returns: {"results": [...], "timings": {"embed_ms": ..., "search_ms": ..., "rerank_ms": ...}}
        """
        if self.bm25 is None or brand_filter:
            self._load_bm25_index(brand_filter)

        if not self.all_documents:
            return {"results": [], "timings": {"embed_ms": 0, "search_ms": 0, "rerank_ms": 0}}

        total_start = _time.perf_counter()

        # --- Embed ---
        embed_start = _time.perf_counter()
        query_embedding = self.embedding_model.encode(query).tolist()
        embed_ms = (_time.perf_counter() - embed_start) * 1000

        # --- Vector search (using pre-computed embedding) ---
        where_filter = None
        if brand_filter:
            where_filter = {"filename": {"$contains": brand_filter.lower()}}

        vs_start = _time.perf_counter()
        vs_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            where=where_filter,
            include=['documents', 'metadatas', 'distances'],
        )
        vector_results = []
        for i in range(len(vs_results['documents'][0])):
            vector_results.append({
                'id': vs_results['ids'][0][i],
                'document': vs_results['documents'][0][i],
                'metadata': vs_results['metadatas'][0][i],
                'score': 1 - vs_results['distances'][0][i],
                'method': 'vector',
            })

        # --- BM25 search ---
        bm25_results = self.bm25_search(query, top_k=20)
        search_ms = (_time.perf_counter() - vs_start) * 1000

        # --- Rerank (RRF) ---
        rerank_start = _time.perf_counter()
        merged_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
        final_results = merged_results[:top_k]
        rerank_ms = (_time.perf_counter() - rerank_start) * 1000

        return {
            "results": final_results,
            "timings": {
                "embed_ms": round(embed_ms, 2),
                "search_ms": round(search_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
            },
        }

    def get_available_brands(self) -> List[str]:
        """Get unique brand names from filenames in the collection."""
        results = self.collection.get(include=['metadatas'])
        filenames = set(meta['filename'] for meta in results['metadatas'])

        brands = set()
        for filename in filenames:
            name = filename.replace('.pdf', '')
            brand = name.split('_')[0].split()[0]
            brands.add(brand)

        return sorted(list(brands))


def main():
    retriever = HybridRetriever()


    print("\nAvailable brands:")
    brands = retriever.get_available_brands()
    for brand in brands:
        print(f"  - {brand}")


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
