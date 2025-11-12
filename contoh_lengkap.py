#!/usr/bin/env python3
"""
Contoh Lengkap: Semua Fitur RAG System
File ini berisi contoh-contoh lengkap yang bisa langsung di-copy-paste
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Import RAG system
from rag_system import LightweightRAG
from ocr_rag_pipeline import OCRRAGPipeline


# ============================================================
# CONTOH 1: RAG System Paling Sederhana
# ============================================================

def contoh_1_basic():
    """Contoh paling sederhana - hanya 5 baris!"""
    print("\n" + "="*60)
    print("CONTOH 1: RAG System Paling Sederhana")
    print("="*60)
    
    # Initialize
    rag = LightweightRAG()
    
    # Add documents
    rag.add_documents(["Python adalah bahasa pemrograman.", "FAISS untuk similarity search."])
    
    # Query
    result = rag.query("Apa itu Python?")
    print(f"\nJawaban: {result['answer']}")


# ============================================================
# CONTOH 2: Dengan Metadata
# ============================================================

def contoh_2_metadata():
    """Contoh dengan metadata untuk tracking sumber"""
    print("\n" + "="*60)
    print("CONTOH 2: RAG dengan Metadata")
    print("="*60)
    
    rag = LightweightRAG(faiss_index_path="./contoh2_index")
    
    # Documents dengan metadata
    documents = [
        "DeepSeek-OCR adalah model OCR yang powerful.",
        "Model ini mendukung berbagai resolusi.",
        "Dirilis pada Oktober 2025."
    ]
    
    metadata = [
        {"source": "intro.txt", "page": 1},
        {"source": "features.txt", "page": 2},
        {"source": "release.txt", "page": 3}
    ]
    
    rag.add_documents(documents, metadata)
    
    # Query
    result = rag.query("Kapan DeepSeek-OCR dirilis?")
    print(f"\nJawaban: {result['answer']}")
    print(f"Sumber: {result['sources']}")


# ============================================================
# CONTOH 3: Custom System Prompt
# ============================================================

def contoh_3_custom_prompt():
    """Contoh dengan custom system prompt untuk domain spesifik"""
    print("\n" + "="*60)
    print("CONTOH 3: Custom System Prompt")
    print("="*60)
    
    rag = LightweightRAG(faiss_index_path="./contoh3_index")
    
    # Add documents tentang keuangan
    documents = [
        "Invoice #001: Total Rp 1.000.000, tanggal 1 Januari 2025",
        "Invoice #002: Total Rp 2.500.000, tanggal 5 Januari 2025",
        "Invoice #003: Total Rp 750.000, tanggal 10 Januari 2025"
    ]
    
    rag.add_documents(documents)
    
    # Custom prompt untuk analisis keuangan
    custom_prompt = """
    Anda adalah asisten untuk analisis dokumen keuangan.
    Fokus pada angka, tanggal, dan informasi penting.
    Berikan jawaban yang akurat dan detail dengan format yang rapi.
    """
    
    result = rag.query(
        "Berapa total semua invoice?",
        system_prompt=custom_prompt
    )
    
    print(f"\nJawaban: {result['answer']}")


# ============================================================
# CONTOH 4: Batch Processing
# ============================================================

def contoh_4_batch():
    """Contoh batch processing untuk banyak dokumen"""
    print("\n" + "="*60)
    print("CONTOH 4: Batch Processing")
    print("="*60)
    
    rag = LightweightRAG(faiss_index_path="./contoh4_index")
    
    # Simulasi banyak dokumen
    all_documents = [
        f"Dokumen {i}: Ini adalah konten dokumen nomor {i}."
        for i in range(1, 101)  # 100 dokumen
    ]
    
    # Process dalam batch
    batch_size = 20
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i+batch_size]
        rag.add_documents(batch)
        print(f"Processed batch {i//batch_size + 1}: {len(batch)} documents")
    
    print(f"\nTotal documents: {len(rag.documents)}")
    
    # Query
    result = rag.query("Berapa banyak dokumen yang ada?")
    print(f"\nJawaban: {result['answer']}")


# ============================================================
# CONTOH 5: Multi-language
# ============================================================

def contoh_5_multilingual():
    """Contoh dengan dokumen multi-bahasa"""
    print("\n" + "="*60)
    print("CONTOH 5: Multi-language Support")
    print("="*60)
    
    # Gunakan multilingual embedding model
    rag = LightweightRAG(
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        faiss_index_path="./contoh5_index"
    )
    
    # Documents dalam berbagai bahasa
    documents = [
        "DeepSeek-OCR is a powerful OCR model.",  # English
        "DeepSeek-OCR adalah model OCR yang powerful.",  # Indonesian
        "DeepSeek-OCR est un modèle OCR puissant.",  # French
        "DeepSeek-OCR es un modelo OCR potente."  # Spanish
    ]
    
    rag.add_documents(documents)
    
    # Query dalam bahasa Indonesia
    result = rag.query("Apa itu DeepSeek-OCR?")
    print(f"\nJawaban: {result['answer']}")


# ============================================================
# CONTOH 6: OCR + RAG (Simple)
# ============================================================

def contoh_6_ocr_simple():
    """Contoh OCR sederhana + RAG"""
    print("\n" + "="*60)
    print("CONTOH 6: OCR Sederhana + RAG")
    print("="*60)
    
    # Initialize pipeline
    pipeline = OCRRAGPipeline(
        use_gpu=False,
        rag_config={
            "llm_provider": "openai",
            "llm_model": "gpt-3.5-turbo",
            "faiss_index_path": "./contoh6_index"
        }
    )
    
    # Check for images
    image_paths = list(Path("./assets").glob("*.jpg")) + list(Path("./assets").glob("*.png"))
    
    if not image_paths:
        print("⚠️  Tidak ada gambar di ./assets")
        print("Tambahkan gambar untuk mencoba OCR")
        return
    
    print(f"Ditemukan {len(image_paths)} gambar")
    
    # Process images (max 2 untuk demo)
    images_to_process = [str(img) for img in image_paths[:2]]
    
    try:
        pipeline.index_images(
            images_to_process,
            use_deepseek_ocr=False  # Simple OCR
        )
        
        # Query
        result = pipeline.query("Apa yang ada di gambar?")
        print(f"\nJawaban: {result['answer']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Install pytesseract untuk OCR sederhana")


# ============================================================
# CONTOH 7: Index dari Folder
# ============================================================

def contoh_7_index_folder():
    """Contoh index semua file dari folder"""
    print("\n" + "="*60)
    print("CONTOH 7: Index dari Folder")
    print("="*60)
    
    rag = LightweightRAG(faiss_index_path="./contoh7_index")
    
    # Simulasi: baca semua .txt files dari folder
    folder_path = Path("./documents")  # Ganti dengan folder Anda
    
    if not folder_path.exists():
        print(f"⚠️  Folder {folder_path} tidak ada")
        print("Buat folder dan tambahkan file .txt untuk mencoba")
        return
    
    txt_files = list(folder_path.glob("*.txt"))
    
    if not txt_files:
        print(f"⚠️  Tidak ada file .txt di {folder_path}")
        return
    
    print(f"Ditemukan {len(txt_files)} file .txt")
    
    # Read and index
    documents = []
    metadata = []
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append(content)
            metadata.append({"source": txt_file.name, "path": str(txt_file)})
    
    rag.add_documents(documents, metadata)
    print(f"✅ Indexed {len(documents)} documents")
    
    # Query
    result = rag.query("Apa isi dokumen?")
    print(f"\nJawaban: {result['answer']}")


# ============================================================
# CONTOH 8: Save & Load Index
# ============================================================

def contoh_8_save_load():
    """Contoh save dan load index"""
    print("\n" + "="*60)
    print("CONTOH 8: Save & Load Index")
    print("="*60)
    
    index_path = "./contoh8_index"
    
    # Create and save
    print("\n1. Create index...")
    rag1 = LightweightRAG(faiss_index_path=index_path)
    rag1.add_documents([
        "Dokumen 1: Python programming",
        "Dokumen 2: Machine learning",
        "Dokumen 3: Deep learning"
    ])
    print(f"✅ Created index with {len(rag1.documents)} documents")
    
    # Load existing index
    print("\n2. Load existing index...")
    rag2 = LightweightRAG(faiss_index_path=index_path)
    print(f"✅ Loaded index with {len(rag2.documents)} documents")
    
    # Query from loaded index
    result = rag2.query("Apa itu machine learning?")
    print(f"\nJawaban: {result['answer']}")
    
    # Clear index
    print("\n3. Clear index...")
    rag2.clear_index()
    print("✅ Index cleared")


# ============================================================
# CONTOH 9: Different LLM Models
# ============================================================

def contoh_9_different_models():
    """Contoh menggunakan model LLM yang berbeda"""
    print("\n" + "="*60)
    print("CONTOH 9: Different LLM Models")
    print("="*60)
    
    # GPT-3.5-turbo (default, murah)
    print("\n1. Using GPT-3.5-turbo...")
    rag1 = LightweightRAG(
        llm_model="gpt-3.5-turbo",
        faiss_index_path="./contoh9_index"
    )
    
    rag1.add_documents(["Python adalah bahasa pemrograman."])
    result1 = rag1.query("Apa itu Python?")
    print(f"GPT-3.5: {result1['answer'][:100]}...")
    
    # GPT-4 (lebih pintar, lebih mahal)
    # Uncomment jika ingin coba GPT-4
    # print("\n2. Using GPT-4...")
    # rag2 = LightweightRAG(
    #     llm_model="gpt-4",
    #     faiss_index_path="./contoh9_index"
    # )
    # result2 = rag2.query("Apa itu Python?")
    # print(f"GPT-4: {result2['answer'][:100]}...")


# ============================================================
# CONTOH 10: Error Handling
# ============================================================

def contoh_10_error_handling():
    """Contoh error handling yang baik"""
    print("\n" + "="*60)
    print("CONTOH 10: Error Handling")
    print("="*60)
    
    try:
        # Initialize
        rag = LightweightRAG(faiss_index_path="./contoh10_index")
        
        # Add documents
        documents = ["Dokumen 1", "Dokumen 2"]
        rag.add_documents(documents)
        
        # Query
        result = rag.query("Test query")
        print(f"✅ Success: {result['answer'][:50]}...")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Tip: Check your .env file and API key")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Tip: Check logs and documentation")


# ============================================================
# MAIN: Run All Examples
# ============================================================

def main():
    """Run all examples"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           📚 CONTOH LENGKAP: RAG SYSTEM 📚                  ║
║                                                              ║
║  File ini berisi 10 contoh lengkap yang bisa langsung       ║
║  di-copy-paste untuk project Anda!                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ ERROR: Set OpenAI API key di file .env terlebih dahulu!")
        print("Baca MULAI_DISINI.md untuk instruksi")
        return
    
    print("✅ API key terdeteksi!\n")
    
    # Menu
    examples = [
        ("RAG System Paling Sederhana", contoh_1_basic),
        ("RAG dengan Metadata", contoh_2_metadata),
        ("Custom System Prompt", contoh_3_custom_prompt),
        ("Batch Processing", contoh_4_batch),
        ("Multi-language Support", contoh_5_multilingual),
        ("OCR Sederhana + RAG", contoh_6_ocr_simple),
        ("Index dari Folder", contoh_7_index_folder),
        ("Save & Load Index", contoh_8_save_load),
        ("Different LLM Models", contoh_9_different_models),
        ("Error Handling", contoh_10_error_handling),
    ]
    
    while True:
        print("\n" + "="*60)
        print("📋 PILIH CONTOH:")
        print("="*60)
        
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i:2d}. {name}")
        
        print(f"{len(examples)+1:2d}. Jalankan Semua Contoh")
        print(" 0. Keluar")
        print("="*60)
        
        try:
            choice = input("\n👉 Pilih (0-11): ").strip()
            
            if choice == "0":
                print("\n👋 Terima kasih!")
                break
            
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(examples):
                examples[choice_num - 1][1]()
            elif choice_num == len(examples) + 1:
                for name, func in examples:
                    func()
                print("\n✅ Semua contoh selesai!")
            else:
                print("❌ Pilihan tidak valid")
                
        except KeyboardInterrupt:
            print("\n\n👋 Terima kasih!")
            break
        except ValueError:
            print("❌ Masukkan angka yang valid")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
