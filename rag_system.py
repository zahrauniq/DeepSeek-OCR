"""
RAG System untuk DeepSeek-OCR
Sistem ringan dengan FAISS untuk indexing dan retrieval
Support multiple LLM providers: OpenAI, DeepSeek, Groq, Ollama
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# FAISS and embeddings
import faiss
from sentence_transformers import SentenceTransformer

# LLM providers
from openai import OpenAI

load_dotenv()


class LightweightRAG:
    """RAG System ringan dengan FAISS"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        faiss_index_path: str = "./faiss_index",
        llm_provider: str = "openai",
        llm_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize RAG system
        
        Args:
            embedding_model: Model untuk embedding (default: all-MiniLM-L6-v2, ringan & cepat)
            faiss_index_path: Path untuk menyimpan FAISS index
            llm_provider: Provider LLM (openai/deepseek/groq/ollama)
            llm_model: Model name
        """
        self.embedding_model_name = embedding_model
        self.faiss_index_path = Path(faiss_index_path)
        self.faiss_index_path.mkdir(exist_ok=True)
        
        # Load embedding model (ringan, ~80MB)
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        
        # LLM setup
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._setup_llm()
        
        # Load existing index if available
        self._load_index()
    
    def _setup_llm(self):
        """Setup LLM client berdasarkan provider"""
        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY tidak ditemukan di .env")
            self.llm_client = OpenAI(api_key=api_key)
            
        elif self.llm_provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY tidak ditemukan di .env")
            # DeepSeek compatible dengan OpenAI API
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
        elif self.llm_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY tidak ditemukan di .env")
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
        elif self.llm_provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # Ollama juga compatible dengan OpenAI API
            self.llm_client = OpenAI(
                api_key="ollama",  # dummy key
                base_url=f"{base_url}/v1"
            )
        else:
            raise ValueError(f"Provider tidak didukung: {self.llm_provider}")
        
        print(f"LLM Provider: {self.llm_provider} | Model: {self.llm_model}")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Tambahkan dokumen ke FAISS index
        
        Args:
            documents: List of text documents
            metadata: Optional metadata untuk setiap dokumen
        """
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to index...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Initialize FAISS index jika belum ada
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add to FAISS
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        print(f"Total documents in index: {len(self.documents)}")
        
        # Auto-save
        self._save_index()
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search dokumen yang relevan
        
        Args:
            query: Query text
            top_k: Jumlah dokumen yang dikembalikan
            
        Returns:
            List of dicts dengan keys: text, metadata, score
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(dist)
                })
        
        return results
    
    def query(self, question: str, top_k: int = 3, system_prompt: Optional[str] = None) -> Dict:
        """
        Query dengan RAG: retrieve + generate
        
        Args:
            question: User question
            top_k: Jumlah dokumen untuk retrieve
            system_prompt: Custom system prompt
            
        Returns:
            Dict dengan keys: answer, sources, retrieved_docs
        """
        # Retrieve relevant documents
        retrieved_docs = self.search(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "Tidak ada dokumen yang ditemukan. Silakan tambahkan dokumen terlebih dahulu.",
                "sources": [],
                "retrieved_docs": []
            }
        
        # Build context from retrieved docs
        context = "\n\n".join([
            f"Dokumen {i+1}:\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.
Gunakan informasi dari dokumen untuk menjawab pertanyaan dengan akurat.
Jika informasi tidak ada dalam dokumen, katakan bahwa Anda tidak menemukan informasi tersebut."""
        
        # Generate answer with LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Berdasarkan dokumen berikut:

{context}

Pertanyaan: {question}

Jawaban:"""}
        ]
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"Error saat generate answer: {str(e)}"
        
        return {
            "answer": answer,
            "sources": [doc["metadata"] for doc in retrieved_docs],
            "retrieved_docs": retrieved_docs
        }
    
    def _save_index(self):
        """Save FAISS index dan metadata"""
        if self.index is None:
            return
        
        # Save FAISS index
        index_file = self.faiss_index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save documents and metadata
        data_file = self.faiss_index_path / "data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Index saved to {self.faiss_index_path}")
    
    def _load_index(self):
        """Load FAISS index dan metadata"""
        index_file = self.faiss_index_path / "index.faiss"
        data_file = self.faiss_index_path / "data.json"
        
        if index_file.exists() and data_file.exists():
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load documents and metadata
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
            
            print(f"Loaded {len(self.documents)} documents from {self.faiss_index_path}")
    
    def clear_index(self):
        """Clear semua data dari index"""
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Delete files
        index_file = self.faiss_index_path / "index.faiss"
        data_file = self.faiss_index_path / "data.json"
        
        if index_file.exists():
            index_file.unlink()
        if data_file.exists():
            data_file.unlink()
        
        print("Index cleared")


def main():
    """Demo penggunaan RAG system"""
    
    # Load config dari environment
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    
    # Initialize RAG
    rag = LightweightRAG(
        embedding_model=embedding_model,
        faiss_index_path=faiss_index_path,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    
    # Demo: Add sample documents
    sample_docs = [
        "DeepSeek-OCR adalah model untuk optical character recognition yang dikembangkan oleh DeepSeek AI.",
        "Model ini mendukung berbagai resolusi: Tiny (512x512), Small (640x640), Base (1024x1024), dan Large (1280x1280).",
        "DeepSeek-OCR dapat mengkonversi dokumen ke markdown dengan prompt: <image>\\n<|grounding|>Convert the document to markdown."
    ]
    
    sample_metadata = [
        {"source": "README.md", "section": "intro"},
        {"source": "README.md", "section": "modes"},
        {"source": "README.md", "section": "prompts"}
    ]
    
    print("\n=== Adding sample documents ===")
    rag.add_documents(sample_docs, sample_metadata)
    
    # Demo: Query
    print("\n=== Querying ===")
    question = "Apa saja resolusi yang didukung DeepSeek-OCR?"
    result = rag.query(question)
    
    print(f"\nPertanyaan: {question}")
    print(f"\nJawaban: {result['answer']}")
    print(f"\nSources: {result['sources']}")


if __name__ == "__main__":
    main()
