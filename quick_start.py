#!/usr/bin/env python3
"""
Quick Start: Contoh Paling Sederhana
Gunakan script ini untuk mulai dengan cepat!
"""

import os
from dotenv import load_dotenv
from rag_system import LightweightRAG

# Load API key dari .env
load_dotenv()

def main():
    print("🚀 Quick Start: RAG System\n")
    
    # 1. Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ ERROR: Silakan set OpenAI API key di file .env")
        print("\n📝 Edit file .env dan ganti:")
        print("   OPENAI_API_KEY=your_openai_api_key_here")
        print("   dengan API key Anda yang sebenarnya")
        return
    
    print("✅ API key terdeteksi!\n")
    
    # 2. Initialize RAG
    print("📚 Inisialisasi RAG system...")
    rag = LightweightRAG(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # 3. Add documents
    print("\n📝 Menambahkan dokumen...")
    documents = [
        "DeepSeek-OCR adalah model OCR yang powerful.",
        "Model ini mendukung resolusi 512x512, 640x640, 1024x1024, dan 1280x1280.",
        "DeepSeek-OCR dapat mengkonversi dokumen ke markdown."
    ]
    
    rag.add_documents(documents)
    print(f"✅ Berhasil menambahkan {len(documents)} dokumen\n")
    
    # 4. Query
    print("❓ Mencoba query...")
    question = "Apa itu DeepSeek-OCR?"
    print(f"   Pertanyaan: {question}\n")
    
    result = rag.query(question)
    print(f"💡 Jawaban:\n   {result['answer']}\n")
    
    print("✅ Selesai! Sistem RAG berjalan dengan baik!")
    print("\n💡 Next steps:")
    print("   - Jalankan: python demo_complete.py (untuk demo lengkap)")
    print("   - Edit script ini untuk menambahkan dokumen Anda sendiri")
    print("   - Lihat README_RAG.md untuk dokumentasi lengkap")


if __name__ == "__main__":
    main()
