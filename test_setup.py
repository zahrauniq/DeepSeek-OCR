#!/usr/bin/env python3
"""
Test Setup: Verifikasi semua dependencies dan konfigurasi
"""

import sys
import os

def test_imports():
    """Test apakah semua library terinstall"""
    print("🔍 Testing imports...")
    
    try:
        import faiss
        print("  ✅ faiss-cpu")
    except ImportError as e:
        print(f"  ❌ faiss-cpu: {e}")
        return False
    
    try:
        import sentence_transformers
        print("  ✅ sentence-transformers")
    except ImportError as e:
        print(f"  ❌ sentence-transformers: {e}")
        return False
    
    try:
        import openai
        print("  ✅ openai")
    except ImportError as e:
        print(f"  ❌ openai: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("  ✅ python-dotenv")
    except ImportError as e:
        print(f"  ❌ python-dotenv: {e}")
        return False
    
    try:
        import tiktoken
        print("  ✅ tiktoken")
    except ImportError as e:
        print(f"  ❌ tiktoken: {e}")
        return False
    
    try:
        from PIL import Image
        print("  ✅ Pillow")
    except ImportError as e:
        print(f"  ❌ Pillow: {e}")
        return False
    
    try:
        import numpy
        print("  ✅ numpy")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        return False
    
    return True


def test_env_file():
    """Test apakah .env file ada dan valid"""
    print("\n🔍 Testing .env file...")
    
    if not os.path.exists(".env"):
        print("  ❌ File .env tidak ditemukan")
        print("     Buat file .env dari .env.example")
        return False
    
    print("  ✅ File .env ditemukan")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ❌ OPENAI_API_KEY tidak ditemukan di .env")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("  ⚠️  OPENAI_API_KEY masih placeholder")
        print("     Ganti dengan API key Anda yang sebenarnya")
        return False
    
    if not api_key.startswith("sk-"):
        print("  ⚠️  OPENAI_API_KEY format tidak valid (harus dimulai dengan 'sk-')")
        return False
    
    print(f"  ✅ OPENAI_API_KEY valid (sk-...{api_key[-4:]})")
    
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    print(f"  ✅ Provider: {provider}")
    print(f"  ✅ Model: {model}")
    
    return True


def test_rag_system():
    """Test apakah RAG system bisa diimport"""
    print("\n🔍 Testing RAG system...")
    
    try:
        from rag_system import LightweightRAG
        print("  ✅ rag_system.py dapat diimport")
        return True
    except Exception as e:
        print(f"  ❌ Error import rag_system: {e}")
        return False


def test_ocr_pipeline():
    """Test apakah OCR pipeline bisa diimport"""
    print("\n🔍 Testing OCR pipeline...")
    
    try:
        from ocr_rag_pipeline import OCRRAGPipeline
        print("  ✅ ocr_rag_pipeline.py dapat diimport")
        return True
    except Exception as e:
        print(f"  ❌ Error import ocr_rag_pipeline: {e}")
        return False


def test_api_connection():
    """Test koneksi ke OpenAI API"""
    print("\n🔍 Testing OpenAI API connection...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("  ⚠️  Skipping (API key belum diset)")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test dengan request sederhana
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        
        print("  ✅ Koneksi ke OpenAI API berhasil!")
        print(f"  ✅ Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error koneksi ke OpenAI API: {e}")
        print("\n  💡 Kemungkinan penyebab:")
        print("     - API key tidak valid")
        print("     - Quota habis (perlu top-up)")
        print("     - Koneksi internet bermasalah")
        return False


def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🧪 TEST SETUP & CONFIGURATION 🧪               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Environment File", test_env_file()))
    results.append(("RAG System", test_rag_system()))
    results.append(("OCR Pipeline", test_ocr_pipeline()))
    results.append(("API Connection", test_api_connection()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for name, result in results:
        if result is True:
            print(f"  ✅ {name}")
        elif result is False:
            print(f"  ❌ {name}")
        else:
            print(f"  ⚠️  {name} (skipped)")
    
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 Semua test passed! Sistem siap digunakan!")
        print("\n💡 Next steps:")
        print("   1. Jalankan: python quick_start.py")
        print("   2. Atau: python demo_complete.py")
        print("   3. Baca: PANDUAN_LENGKAP.md")
        return 0
    else:
        print("\n⚠️  Ada test yang gagal. Silakan perbaiki error di atas.")
        print("\n💡 Tips:")
        print("   - Install dependencies: pip install -r rag_requirements.txt")
        print("   - Set API key di file .env")
        print("   - Baca PANDUAN_LENGKAP.md untuk bantuan")
        return 1


if __name__ == "__main__":
    sys.exit(main())
