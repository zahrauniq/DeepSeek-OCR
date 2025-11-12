#!/usr/bin/env python3
"""
Demo Lengkap: OCR + RAG System
Script ini menunjukkan cara menggunakan sistem OCR + RAG dari awal sampai akhir
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG system
from rag_system import LightweightRAG
from ocr_rag_pipeline import OCRRAGPipeline


def check_api_key():
    """Check apakah API key sudah diset"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ ERROR: OpenAI API key belum diset!")
        print("\n📝 Cara setup:")
        print("1. Buka file .env")
        print("2. Ganti 'your_openai_api_key_here' dengan API key Anda")
        print("3. Simpan file dan jalankan lagi script ini")
        print("\n💡 Cara mendapatkan API key:")
        print("   - Kunjungi: https://platform.openai.com/api-keys")
        print("   - Login/daftar")
        print("   - Klik 'Create new secret key'")
        print("   - Copy dan paste ke file .env")
        return False
    return True


def demo_1_basic_rag():
    """Demo 1: RAG System Dasar (tanpa OCR)"""
    print("\n" + "="*60)
    print("📚 DEMO 1: RAG System Dasar")
    print("="*60)
    
    # Initialize RAG
    print("\n1️⃣ Inisialisasi RAG system...")
    rag = LightweightRAG(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        faiss_index_path="./demo_faiss_index"
    )
    
    # Add sample documents
    print("\n2️⃣ Menambahkan dokumen sample...")
    documents = [
        "DeepSeek-OCR adalah model OCR yang dikembangkan oleh DeepSeek AI untuk optical character recognition.",
        "Model ini mendukung berbagai resolusi: Tiny (512x512 dengan 64 vision tokens), Small (640x640 dengan 100 tokens), Base (1024x1024 dengan 256 tokens), dan Large (1280x1280 dengan 400 tokens).",
        "DeepSeek-OCR dapat mengkonversi dokumen ke markdown menggunakan prompt: <image>\\n<|grounding|>Convert the document to markdown.",
        "Model ini juga mendukung dynamic resolution dengan mode Gundam: n×640×640 + 1×1024×1024.",
        "DeepSeek-OCR dirilis pada 20 Oktober 2025 dan sekarang didukung oleh vLLM upstream."
    ]
    
    metadata = [
        {"source": "README.md", "section": "intro"},
        {"source": "README.md", "section": "resolutions"},
        {"source": "README.md", "section": "prompts"},
        {"source": "README.md", "section": "modes"},
        {"source": "README.md", "section": "release"}
    ]
    
    rag.add_documents(documents, metadata)
    print(f"✅ Berhasil menambahkan {len(documents)} dokumen")
    
    # Query examples
    print("\n3️⃣ Mencoba beberapa query...")
    questions = [
        "Apa itu DeepSeek-OCR?",
        "Resolusi apa saja yang didukung?",
        "Bagaimana cara mengkonversi dokumen ke markdown?",
        "Kapan DeepSeek-OCR dirilis?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*60}")
        print(f"❓ Pertanyaan {i}: {question}")
        print(f"{'─'*60}")
        
        result = rag.query(question, top_k=2)
        print(f"\n💡 Jawaban:\n{result['answer']}")
        
        if result['sources']:
            print(f"\n📄 Sumber: {result['sources']}")
    
    print("\n✅ Demo 1 selesai!")
    return rag


def demo_2_ocr_simple():
    """Demo 2: OCR Sederhana + RAG"""
    print("\n" + "="*60)
    print("🖼️  DEMO 2: OCR Sederhana + RAG")
    print("="*60)
    
    # Initialize pipeline
    print("\n1️⃣ Inisialisasi OCR + RAG pipeline...")
    pipeline = OCRRAGPipeline(
        use_gpu=False,  # Tidak perlu GPU
        rag_config={
            "llm_provider": "openai",
            "llm_model": "gpt-3.5-turbo",
            "faiss_index_path": "./demo_ocr_faiss_index"
        }
    )
    
    # Check for images
    print("\n2️⃣ Mencari gambar di folder ./assets...")
    image_paths = list(Path("./assets").glob("*.jpg")) + list(Path("./assets").glob("*.png"))
    
    if not image_paths:
        print("⚠️  Tidak ada gambar di folder ./assets")
        print("💡 Tip: Tambahkan gambar (.jpg atau .png) ke folder ./assets untuk mencoba OCR")
        return None
    
    print(f"✅ Ditemukan {len(image_paths)} gambar")
    
    # Process images (max 3 untuk demo)
    images_to_process = [str(img) for img in image_paths[:3]]
    print(f"\n3️⃣ Memproses {len(images_to_process)} gambar dengan OCR sederhana...")
    print("⏳ Ini mungkin memakan waktu beberapa detik...")
    
    try:
        pipeline.index_images(
            images_to_process,
            use_deepseek_ocr=False  # Gunakan OCR sederhana
        )
        
        # Query
        print("\n4️⃣ Mencoba query...")
        questions = [
            "Apa yang ada di gambar?",
            "Jelaskan isi dari gambar-gambar tersebut"
        ]
        
        for question in questions:
            print(f"\n{'─'*60}")
            print(f"❓ Pertanyaan: {question}")
            print(f"{'─'*60}")
            
            result = pipeline.query(question, top_k=2)
            print(f"\n💡 Jawaban:\n{result['answer']}")
        
        print("\n✅ Demo 2 selesai!")
        return pipeline
        
    except Exception as e:
        print(f"\n❌ Error saat OCR: {e}")
        print("\n💡 Tip: Pastikan pytesseract terinstall:")
        print("   pip install pytesseract")
        print("   Atau gunakan DeepSeek-OCR untuk hasil terbaik (butuh GPU)")
        return None


def demo_3_custom_documents():
    """Demo 3: Tambahkan Dokumen Custom"""
    print("\n" + "="*60)
    print("📝 DEMO 3: Tambahkan Dokumen Custom Anda")
    print("="*60)
    
    # Initialize RAG
    rag = LightweightRAG(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        faiss_index_path="./demo_custom_faiss_index"
    )
    
    print("\n💡 Anda bisa menambahkan dokumen custom Anda sendiri!")
    print("\nContoh code:")
    print("""
    # Tambahkan dokumen
    my_documents = [
        "Dokumen pertama saya...",
        "Dokumen kedua saya...",
    ]
    
    rag.add_documents(my_documents)
    
    # Query
    result = rag.query("Pertanyaan saya?")
    print(result['answer'])
    """)
    
    # Demo dengan dokumen custom
    print("\n1️⃣ Menambahkan dokumen custom...")
    custom_docs = [
        "Python adalah bahasa pemrograman yang populer untuk AI dan machine learning.",
        "FAISS adalah library untuk similarity search yang dikembangkan oleh Facebook AI Research.",
        "RAG (Retrieval-Augmented Generation) menggabungkan retrieval dan generation untuk menjawab pertanyaan.",
        "OpenAI GPT-3.5-turbo adalah model language yang powerful untuk berbagai tugas NLP."
    ]
    
    rag.add_documents(custom_docs)
    print(f"✅ Berhasil menambahkan {len(custom_docs)} dokumen custom")
    
    # Query
    print("\n2️⃣ Query dokumen custom...")
    question = "Apa itu RAG dan bagaimana cara kerjanya?"
    print(f"\n❓ Pertanyaan: {question}")
    
    result = rag.query(question)
    print(f"\n💡 Jawaban:\n{result['answer']}")
    
    print("\n✅ Demo 3 selesai!")
    return rag


def demo_4_interactive():
    """Demo 4: Mode Interaktif"""
    print("\n" + "="*60)
    print("💬 DEMO 4: Mode Interaktif")
    print("="*60)
    
    # Initialize RAG dengan dokumen sample
    rag = LightweightRAG(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        faiss_index_path="./demo_interactive_faiss_index"
    )
    
    # Add sample documents
    print("\n1️⃣ Memuat dokumen sample...")
    documents = [
        "DeepSeek-OCR adalah model OCR yang powerful untuk optical character recognition.",
        "Model ini mendukung berbagai resolusi dari 512x512 hingga 1280x1280 pixels.",
        "Anda bisa menggunakan DeepSeek-OCR dengan vLLM atau Transformers.",
        "Model ini dirilis pada Oktober 2025 dan open source.",
        "DeepSeek-OCR sangat cocok untuk konversi dokumen ke markdown."
    ]
    
    rag.add_documents(documents)
    print(f"✅ Berhasil memuat {len(documents)} dokumen")
    
    print("\n2️⃣ Mode Interaktif - Tanya apa saja!")
    print("💡 Ketik 'exit' atau 'quit' untuk keluar")
    print("="*60)
    
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()
            
            if question.lower() in ['exit', 'quit', 'keluar', 'q']:
                print("\n👋 Terima kasih! Sampai jumpa!")
                break
            
            if not question:
                continue
            
            print("\n⏳ Memproses...")
            result = rag.query(question)
            print(f"\n💡 Jawaban:\n{result['answer']}")
            print(f"\n{'─'*60}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Terima kasih! Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def main():
    """Main function"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🚀 DEMO LENGKAP: OCR + RAG SYSTEM 🚀                 ║
║                                                              ║
║  Script ini mendemonstrasikan penggunaan sistem OCR + RAG   ║
║  dari awal sampai akhir dengan OpenAI API                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    print("\n✅ API key terdeteksi!")
    print(f"📦 Provider: {os.getenv('LLM_PROVIDER', 'openai')}")
    print(f"🤖 Model: {os.getenv('LLM_MODEL', 'gpt-3.5-turbo')}")
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("📋 PILIH DEMO:")
        print("="*60)
        print("1. Demo RAG System Dasar (tanpa OCR)")
        print("2. Demo OCR Sederhana + RAG")
        print("3. Demo Dokumen Custom")
        print("4. Mode Interaktif (Tanya Jawab)")
        print("5. Jalankan Semua Demo")
        print("0. Keluar")
        print("="*60)
        
        try:
            choice = input("\n👉 Pilih (0-5): ").strip()
            
            if choice == "0":
                print("\n👋 Terima kasih! Sampai jumpa!")
                break
            elif choice == "1":
                demo_1_basic_rag()
            elif choice == "2":
                demo_2_ocr_simple()
            elif choice == "3":
                demo_3_custom_documents()
            elif choice == "4":
                demo_4_interactive()
            elif choice == "5":
                demo_1_basic_rag()
                demo_2_ocr_simple()
                demo_3_custom_documents()
                print("\n✅ Semua demo selesai!")
            else:
                print("❌ Pilihan tidak valid. Silakan pilih 0-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Terima kasih! Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
