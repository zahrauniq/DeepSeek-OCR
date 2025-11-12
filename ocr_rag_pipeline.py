"""
Pipeline lengkap: OCR dengan DeepSeek-OCR + RAG dengan FAISS
"""

import os
from pathlib import Path
from typing import List, Optional
import json
from PIL import Image

# Import RAG system
from rag_system import LightweightRAG


class OCRRAGPipeline:
    """Pipeline untuk OCR + RAG"""
    
    def __init__(
        self,
        ocr_model_name: str = "deepseek-ai/DeepSeek-OCR",
        use_gpu: bool = False,
        rag_config: Optional[dict] = None
    ):
        """
        Initialize OCR + RAG pipeline
        
        Args:
            ocr_model_name: DeepSeek-OCR model name
            use_gpu: Gunakan GPU untuk OCR (butuh CUDA)
            rag_config: Config untuk RAG system
        """
        self.ocr_model_name = ocr_model_name
        self.use_gpu = use_gpu
        
        # Initialize RAG
        rag_config = rag_config or {}
        self.rag = LightweightRAG(**rag_config)
        
        # OCR model akan di-load on-demand untuk hemat memory
        self.ocr_model = None
        self.ocr_tokenizer = None
    
    def _load_ocr_model(self):
        """Load OCR model (lazy loading)"""
        if self.ocr_model is not None:
            return
        
        print("Loading DeepSeek-OCR model...")
        print("⚠️  Warning: Model ini besar (~10GB). Pastikan Anda punya cukup RAM/VRAM.")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self.ocr_tokenizer = AutoTokenizer.from_pretrained(
                self.ocr_model_name,
                trust_remote_code=True
            )
            
            if self.use_gpu and torch.cuda.is_available():
                print("Using GPU for OCR")
                self.ocr_model = AutoModel.from_pretrained(
                    self.ocr_model_name,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True
                )
                self.ocr_model = self.ocr_model.eval().cuda().to(torch.bfloat16)
            else:
                print("Using CPU for OCR (akan lambat)")
                self.ocr_model = AutoModel.from_pretrained(
                    self.ocr_model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                )
                self.ocr_model = self.ocr_model.eval()
            
            print("OCR model loaded successfully")
            
        except Exception as e:
            print(f"Error loading OCR model: {e}")
            print("\nTips:")
            print("1. Pastikan sudah install: pip install -r rag_requirements.txt")
            print("2. Jika error memory, coba gunakan CPU mode (use_gpu=False)")
            print("3. Jika masih error, gunakan OCR alternatif (lihat method process_images_simple)")
            raise
    
    def process_image_with_ocr(
        self,
        image_path: str,
        prompt: str = "<image>\\n<|grounding|>Convert the document to markdown.",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True
    ) -> str:
        """
        Process image dengan DeepSeek-OCR
        
        Args:
            image_path: Path ke image
            prompt: OCR prompt
            base_size: Base resolution
            image_size: Image resolution
            crop_mode: Use crop mode (Gundam mode)
            
        Returns:
            Extracted text
        """
        self._load_ocr_model()
        
        print(f"Processing image: {image_path}")
        
        # Create temp output dir
        output_path = "./temp_ocr_output"
        os.makedirs(output_path, exist_ok=True)
        
        # Run OCR
        result = self.ocr_model.infer(
            self.ocr_tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,
            test_compress=True
        )
        
        return result
    
    def process_images_simple(self, image_paths: List[str]) -> List[str]:
        """
        Process images dengan OCR sederhana (fallback jika DeepSeek-OCR terlalu berat)
        Menggunakan pytesseract atau easyocr sebagai alternatif
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of extracted texts
        """
        print("⚠️  Using simple OCR (pytesseract/easyocr)")
        print("Untuk hasil terbaik, gunakan DeepSeek-OCR dengan GPU")
        
        results = []
        
        try:
            # Try pytesseract first (lebih ringan)
            import pytesseract
            
            for img_path in image_paths:
                img = Image.open(img_path)
                text = pytesseract.image_to_string(img)
                results.append(text)
                print(f"Processed: {img_path}")
            
        except ImportError:
            print("pytesseract not found. Install: pip install pytesseract")
            print("Or use DeepSeek-OCR for better results")
            
            # Fallback: return dummy text
            for img_path in image_paths:
                results.append(f"[OCR placeholder for {img_path}]")
        
        return results
    
    def index_images(
        self,
        image_paths: List[str],
        use_deepseek_ocr: bool = False,
        metadata: Optional[List[dict]] = None
    ):
        """
        Index images: OCR -> extract text -> add to RAG
        
        Args:
            image_paths: List of image paths
            use_deepseek_ocr: Use DeepSeek-OCR (butuh GPU) atau simple OCR
            metadata: Optional metadata untuk setiap image
        """
        print(f"\n=== Indexing {len(image_paths)} images ===")
        
        # Extract text from images
        if use_deepseek_ocr:
            texts = [
                self.process_image_with_ocr(img_path)
                for img_path in image_paths
            ]
        else:
            texts = self.process_images_simple(image_paths)
        
        # Prepare metadata
        if metadata is None:
            metadata = [
                {"source": str(Path(img_path).name), "type": "image"}
                for img_path in image_paths
            ]
        
        # Add to RAG
        self.rag.add_documents(texts, metadata)
        
        print(f"✓ Indexed {len(texts)} documents")
    
    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Query RAG system
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with answer, sources, retrieved_docs
        """
        return self.rag.query(question, top_k=top_k)
    
    def save_state(self, filepath: str = "./pipeline_state.json"):
        """Save pipeline state"""
        state = {
            "ocr_model_name": self.ocr_model_name,
            "use_gpu": self.use_gpu,
            "num_documents": len(self.rag.documents)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Pipeline state saved to {filepath}")


def main():
    """Demo pipeline"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize pipeline
    pipeline = OCRRAGPipeline(
        use_gpu=False,  # Set True jika punya GPU
        rag_config={
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "llm_model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            "faiss_index_path": "./faiss_index"
        }
    )
    
    # Demo: Index sample images (jika ada)
    sample_images = list(Path("./assets").glob("*.jpg")) + list(Path("./assets").glob("*.png"))
    
    if sample_images:
        print(f"\nFound {len(sample_images)} images in ./assets")
        
        # Gunakan simple OCR untuk demo (tidak perlu GPU)
        pipeline.index_images(
            [str(img) for img in sample_images[:3]],  # Process 3 images saja
            use_deepseek_ocr=False  # Set True untuk hasil terbaik
        )
        
        # Query
        print("\n=== Querying ===")
        result = pipeline.query("Apa yang ada di gambar?")
        print(f"\nJawaban: {result['answer']}")
    else:
        print("\nTidak ada gambar di ./assets untuk demo")
        print("Tambahkan gambar ke folder ./assets atau gunakan path lain")


if __name__ == "__main__":
    main()
