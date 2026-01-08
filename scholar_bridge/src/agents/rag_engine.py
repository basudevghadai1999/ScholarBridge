import os
import requests
import fitz # PyMuPDF
import chromadb
import google.generativeai as genai
from typing import List, Dict, Any

class RagEngine:
    """
    Implements a 15-Step RAG Pipeline:
    1. Source Identification | 2. Ingestion | 3. Preprocessing | 4. Chunking
    5. Embedding | 6. Metadata | 7. Storage | 8. Indexing
    9. Query Handling | 10. Query Embedding | 11. Similarity Search
    12. Context Filtering | 13. Prompting | 14. Completion | 15. Post-processing
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        genai.configure(api_key=self.api_key)
        self.client = chromadb.Client()
        self.ef = self._google_embedding_function

    # Step 5 & 10: Text/Query Embedding
    def _google_embedding_function(self, input: List[str]) -> List[List[float]]:
        # Using a model optimized for retrieval
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=input,
            task_type="retrieval_document"
        )
        return result['embedding']

    # Step 1: Source Identification (handled by caller, but we validate here)
    # Step 2: Data Ingestion
    def download_pdf(self, url: str) -> str:
        """Step 2: Downloads the raw data (PDF)."""
        try:
            if "arxiv.org/abs/" in url:
                url = url.replace("abs", "pdf")
            if not url.endswith(".pdf"):
                url += ".pdf"
                
            print(f"[Step 2] Ingesting Data from {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filename = f"temp_{os.path.basename(url)}"
            if not filename.endswith(".pdf"): filename = "temp_paper.pdf"
            path = os.path.join("scholar_bridge/data", filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, "wb") as f:
                f.write(response.content)
            return path
        except Exception as e:
            print(f"Error in Step 2: {e}")
            return ""


    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Recursively splits text by separators to keep semantic meaning.
        Priority: Paragraphs (\n\n) -> Sentences (\n) -> Words ( ) -> Characters
        """
        separators = ["\n\n", "\n", " ", ""]
        return self._recursive_split(text, separators, chunk_size, overlap)

    def _recursive_split(self, text: str, separators: List[str], chunk_size: int, overlap: int) -> List[str]:
        final_chunks = []
        
        # Determine which separator to use
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i+1:]
                break
                
        # Split text by the chosen separator
        splits = [s for s in text.split(separator) if s] if separator else list(text)
        
        # Now merge splits into chunks
        good_splits = []
        separator_len = len(separator)
        
        current_chunk_splits = []
        current_len = 0
        
        for split in splits:
            split_len = len(split)
            
            # If a single split is too big, drill down recursively
            if split_len > chunk_size:
                if current_chunk_splits:
                    # Flush current buffer
                    good_splits.append(separator.join(current_chunk_splits))
                    current_chunk_splits = []
                    current_len = 0
                
                # Recursively chunk this big split
                if new_separators:
                    sub_chunks = self._recursive_split(split, new_separators, chunk_size, overlap)
                    good_splits.extend(sub_chunks)
                else:
                    # Hard cut if no more separators
                    for i in range(0, len(split), chunk_size - overlap):
                        good_splits.append(split[i:i+chunk_size])
                continue

            # Check if adding this split exceeds chunk_size
            if current_len + split_len + (separator_len if current_chunk_splits else 0) > chunk_size:
                # Chunk is full, add to list
                good_splits.append(separator.join(current_chunk_splits))
                
                # Handle Overlap: Keep some previous splits for context
                # Backtrack to find how much to keep
                overlap_splits = []
                overlap_len = 0
                for s in reversed(current_chunk_splits):
                    if overlap_len + len(s) + separator_len <= overlap:
                        overlap_splits.insert(0, s)
                        overlap_len += len(s) + separator_len
                    else:
                        break
                        
                current_chunk_splits = list(overlap_splits)
                current_len = overlap_len
                
            current_chunk_splits.append(split)
            current_len += split_len + (separator_len if len(current_chunk_splits) > 1 else 0)
            
        # Flush remainder
        if current_chunk_splits:
            good_splits.append(separator.join(current_chunk_splits))
            
        return good_splits

    def ingest_paper(self, pdf_path: str) -> str:
        """Orchestrates Steps 3-8."""
        if not pdf_path or not os.path.exists(pdf_path): return ""

        try:
            # Step 3: Preprocessing (Read & Clean)
            print("[Step 3] Preprocessing PDF...")
            doc = fitz.open(pdf_path)
            
            chunks = []
            metadatas = []
            ids = []
            
            # Configuration
            chunk_size = 1000
            overlap = 200 
            
            print("[Step 4] Chunking Document (Recursive w/ Overlap)...")
            global_idx = 0
            
            # Process entire doc text or page by page?
            # Recursive splitter works best on larger contexts, but page-by-page allows better metadata (Page Numbers).
            # We will maintain page-by-page but use recursive splitter on each page content.
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                # Basic cleaning
                text = text.replace('-\n', '').replace('\n', ' ')
                
                # Use Recursive Splitter
                page_chunks = self._chunk_text(text, chunk_size, overlap)
                
                for chunk_text in page_chunks:
                    chunks.append(chunk_text)
                    
                    # Step 6: Metadata Tagging
                    metadatas.append({
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "chunk_id": global_idx
                    })
                    ids.append(str(global_idx))
                    global_idx += 1

            # Step 7: Vector Storage & Step 8: Index Optimization (Chroma handles HNSW)
            print(f"[Step 7] Storing {len(chunks)} Vectors in ChromaDB...")
            collection_name = f"paper_{os.path.basename(pdf_path).replace('.pdf', '').replace('.', '_')}"
            try: self.client.delete_collection(collection_name)
            except: pass
            
            collection = self.client.create_collection(name=collection_name)
            
            # Step 5: Embedding (Batch)
            embeddings = [self._google_embedding_function([c])[0] for c in chunks]
            
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            return collection_name
            
        except Exception as e:
            print(f"Error in Ingestion: {e}")
            return ""

    # Step 9-12: Retrieval
    def query(self, collection_name: str, query_text: str, n_results=5) -> str:
        """Orchestrates Steps 9-12."""
        if not collection_name: return ""
            
        try:
            collection = self.client.get_collection(collection_name)
            
            # Step 9: Query Handling (Input)
            # Step 10: Query Embedding
            # Step 11: Top K Similarity Search
            print(f"[Step 11] Searching for '{query_text}'...")
            query_embed = self._google_embedding_function([query_text])[0]
            
            results = collection.query(
                query_embeddings=[query_embed],
                n_results=n_results
            )
            
            # Step 12: Context Filtering (Thresholding)
            # Chroma returns distances (L2). Lower is better. 
            # Let's say we filter out very far chunks if needed. For now, we utilize all top-k.
            
            valid_docs = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    # Add Metadata context to the text
                    valid_docs.append(f"[Page {meta['page']}] {doc}")
            
            return "\n\n".join(valid_docs)
            
        except Exception as e:
            print(f"Error in Querying: {e}")
            return ""

    # Steps 13-15 are handled by the Agents (ReactResearcher / Writer)
    # Step 13: Prompt Construction -> ReactResearcher
    # Step 14: LLM Completion -> GeminiClient
    # Step 15: Post-processing -> WriterAgent

