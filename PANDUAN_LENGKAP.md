# 📚 Panduan Lengkap: OCR + RAG System

Panduan ini akan membantu Anda menggunakan sistem OCR + RAG dari awal sampai akhir.

## 🎯 Apa yang Akan Anda Pelajari?

1. ✅ Setup API key OpenAI
2. ✅ Install dependencies
3. ✅ Menggunakan RAG system
4. ✅ Menambahkan dokumen custom
5. ✅ Query dan mendapatkan jawaban

---

## 📦 Step 1: Setup API Key

### 1.1 Dapatkan API Key OpenAI

1. Kunjungi: https://platform.openai.com/api-keys
2. Login atau daftar akun baru
3. Klik **"Create new secret key"**
4. Copy API key yang muncul (format: `sk-...`)
5. **PENTING**: Simpan API key ini, tidak bisa dilihat lagi!

### 1.2 Isi Saldo (Jika Perlu)

- Kunjungi: https://platform.openai.com/account/billing
- Minimum top-up: $5
- Harga GPT-3.5-turbo: ~$0.002 per 1K tokens (sangat murah!)

### 1.3 Set API Key di File .env

Edit file `.env` dan ganti:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Dengan API key Anda yang sebenarnya:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
```

**Contoh lengkap file .env:**

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo

# FAISS Settings
FAISS_INDEX_PATH=./faiss_index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🔧 Step 2: Install Dependencies

Dependencies sudah terinstall otomatis, tapi jika perlu install manual:

```bash
pip install faiss-cpu sentence-transformers openai python-dotenv tiktoken pytesseract pillow numpy
```

**Verifikasi instalasi:**

```bash
python3 -c "import faiss; import sentence_transformers; import openai; print('✅ Semua library terinstall!')"
```

---

## 🚀 Step 3: Quick Start (5 Menit)

### 3.1 Test Cepat

Jalankan script quick start:

```bash
python quick_start.py
```

**Output yang diharapkan:**

```
🚀 Quick Start: RAG System

✅ API key terdeteksi!

📚 Inisialisasi RAG system...
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
LLM Provider: openai | Model: gpt-3.5-turbo

📝 Menambahkan dokumen...
Adding 3 documents to index...
✅ Berhasil menambahkan 3 dokumen

❓ Mencoba query...
   Pertanyaan: Apa itu DeepSeek-OCR?

💡 Jawaban:
   DeepSeek-OCR adalah model OCR yang powerful...

✅ Selesai! Sistem RAG berjalan dengan baik!
```

### 3.2 Jika Ada Error

**Error: API key tidak valid**
```
❌ Error: Incorrect API key provided
```
→ Pastikan API key di .env benar dan dimulai dengan `sk-`

**Error: Insufficient quota**
```
❌ Error: You exceeded your current quota
```
→ Isi saldo di https://platform.openai.com/account/billing

**Error: Module not found**
```
❌ ModuleNotFoundError: No module named 'faiss'
```
→ Install ulang: `pip install faiss-cpu`

---

## 📖 Step 4: Demo Lengkap

Jalankan demo interaktif:

```bash
python demo_complete.py
```

### Menu Demo:

```
📋 PILIH DEMO:
1. Demo RAG System Dasar (tanpa OCR)
2. Demo OCR Sederhana + RAG
3. Demo Dokumen Custom
4. Mode Interaktif (Tanya Jawab)
5. Jalankan Semua Demo
0. Keluar
```

**Rekomendasi:**
- Mulai dengan **Demo 1** untuk memahami dasar RAG
- Coba **Demo 4** untuk mode interaktif (tanya jawab)
- Gunakan **Demo 3** untuk dokumen custom Anda

---

## 💻 Step 5: Gunakan di Code Anda

### 5.1 Contoh Sederhana

```python
from dotenv import load_dotenv
from rag_system import LightweightRAG

# Load API key
load_dotenv()

# Initialize RAG
rag = LightweightRAG(
    llm_provider="openai",
    llm_model="gpt-3.5-turbo"
)

# Add documents
documents = [
    "Python adalah bahasa pemrograman yang populer.",
    "FAISS adalah library untuk similarity search.",
    "RAG menggabungkan retrieval dan generation."
]

rag.add_documents(documents)

# Query
result = rag.query("Apa itu RAG?")
print(result['answer'])
```

### 5.2 Dengan Metadata

```python
# Add documents dengan metadata
documents = [
    "Dokumen pertama...",
    "Dokumen kedua..."
]

metadata = [
    {"source": "file1.txt", "date": "2025-01-01"},
    {"source": "file2.txt", "date": "2025-01-02"}
]

rag.add_documents(documents, metadata)

# Query dengan melihat sumber
result = rag.query("Pertanyaan saya?")
print(f"Jawaban: {result['answer']}")
print(f"Sumber: {result['sources']}")
```

### 5.3 Custom System Prompt

```python
# Custom prompt untuk domain spesifik
custom_prompt = """
Anda adalah asisten untuk analisis dokumen keuangan.
Fokus pada angka, tanggal, dan informasi penting.
Berikan jawaban yang akurat dan detail.
"""

result = rag.query(
    "Berapa total invoice?",
    system_prompt=custom_prompt
)
```

---

## 🖼️ Step 6: OCR + RAG (Untuk Gambar)

### 6.1 OCR Sederhana (Tanpa GPU)

```python
from ocr_rag_pipeline import OCRRAGPipeline

# Initialize pipeline
pipeline = OCRRAGPipeline(
    use_gpu=False,  # Tidak perlu GPU
    rag_config={
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo"
    }
)

# Process images
image_paths = ["image1.jpg", "image2.jpg"]
pipeline.index_images(
    image_paths,
    use_deepseek_ocr=False  # Gunakan OCR sederhana
)

# Query
result = pipeline.query("Apa isi dokumen?")
print(result['answer'])
```

### 6.2 DeepSeek-OCR (Hasil Terbaik, Butuh GPU)

```python
# Untuk hasil OCR terbaik
pipeline = OCRRAGPipeline(
    use_gpu=True,  # Butuh GPU
    rag_config={
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo"
    }
)

# Process dengan DeepSeek-OCR
pipeline.index_images(
    image_paths,
    use_deepseek_ocr=True  # Hasil terbaik!
)
```

---

## 🎓 Step 7: Use Cases Praktis

### 7.1 Index Folder Dokumen

```python
from pathlib import Path

# Get all images dari folder
images = list(Path("./documents").glob("*.jpg"))
images += list(Path("./documents").glob("*.png"))

# Index semua
pipeline.index_images([str(img) for img in images])

# Query
result = pipeline.query("Cari informasi tentang invoice")
```

### 7.2 Batch Processing

```python
# Process dalam batch untuk dataset besar
all_images = [...]  # List semua gambar

batch_size = 10
for i in range(0, len(all_images), batch_size):
    batch = all_images[i:i+batch_size]
    pipeline.index_images(batch)
    print(f"Processed batch {i//batch_size + 1}")
```

### 7.3 Multi-language Support

```python
# Gunakan multilingual embedding
rag = LightweightRAG(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Add documents dalam berbagai bahasa
docs = [
    "DeepSeek-OCR is a powerful OCR model",
    "DeepSeek-OCR adalah model OCR yang powerful",
    "DeepSeek-OCR est un modèle OCR puissant"
]

rag.add_documents(docs)
```

---

## 🔍 Step 8: Tips & Tricks

### 8.1 Hemat Biaya OpenAI

```python
# Gunakan model yang lebih murah
rag = LightweightRAG(
    llm_model="gpt-3.5-turbo"  # Lebih murah dari GPT-4
)

# Limit retrieved documents
result = rag.query(question, top_k=2)  # Hanya ambil 2 dokumen
```

### 8.2 Improve Accuracy

```python
# Gunakan embedding model yang lebih baik
rag = LightweightRAG(
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

# Retrieve lebih banyak dokumen
result = rag.query(question, top_k=5)
```

### 8.3 Save & Load Index

```python
# Index otomatis tersimpan di faiss_index_path
# Untuk load ulang, cukup initialize dengan path yang sama

rag = LightweightRAG(
    faiss_index_path="./my_saved_index"
)
# Index akan otomatis di-load jika ada

# Clear index jika perlu
rag.clear_index()
```

---

## 🐛 Troubleshooting

### Problem 1: API Key Error

**Error:**
```
ValueError: OPENAI_API_KEY tidak ditemukan di .env
```

**Solusi:**
1. Pastikan file `.env` ada di folder yang sama
2. Check isi file `.env` dengan: `cat .env`
3. Pastikan format benar: `OPENAI_API_KEY=sk-...`
4. Tidak ada spasi sebelum/sesudah `=`

### Problem 2: Quota Exceeded

**Error:**
```
Error: You exceeded your current quota
```

**Solusi:**
1. Isi saldo di: https://platform.openai.com/account/billing
2. Atau gunakan provider gratis (Groq/Ollama) - lihat README_RAG.md

### Problem 3: Slow Performance

**Masalah:** Query lambat

**Solusi:**
1. Gunakan embedding model yang lebih kecil:
   ```python
   embedding_model="sentence-transformers/all-MiniLM-L6-v2"
   ```
2. Reduce top_k:
   ```python
   result = rag.query(question, top_k=2)
   ```
3. Gunakan GPU jika tersedia

### Problem 4: OCR Error

**Error:**
```
pytesseract not found
```

**Solusi:**
```bash
# Install pytesseract
pip install pytesseract

# Linux: install tesseract
sudo apt-get install tesseract-ocr

# Mac:
brew install tesseract
```

---

## 📊 Performa & Biaya

### Biaya OpenAI (GPT-3.5-turbo)

| Penggunaan | Tokens | Biaya |
|------------|--------|-------|
| 1 query sederhana | ~500 | $0.001 |
| 100 queries | ~50K | $0.10 |
| 1000 queries | ~500K | $1.00 |

**Kesimpulan:** Sangat murah! $5 bisa untuk ribuan queries.

### Memory Usage

| Component | Memory |
|-----------|--------|
| Embedding model | ~80MB |
| FAISS index (1000 docs) | ~1.5MB |
| Total | ~100MB |

**Kesimpulan:** Sangat ringan!

---

## 🎯 Next Steps

1. ✅ **Sudah berjalan?** Coba tambahkan dokumen Anda sendiri!
2. 📚 **Baca dokumentasi:** Lihat `README_RAG.md` untuk detail
3. 🖼️ **Coba OCR:** Tambahkan gambar ke `./assets` dan jalankan Demo 2
4. 🚀 **Production:** Deploy ke server atau cloud

---

## 📞 Bantuan & Support

- **GitHub Issues:** https://github.com/deepseek-ai/DeepSeek-OCR/issues
- **OpenAI Docs:** https://platform.openai.com/docs
- **FAISS Docs:** https://github.com/facebookresearch/faiss

---

## 📄 License

MIT License - Bebas digunakan untuk project apapun!

---

**Selamat mencoba! 🎉**

Jika ada pertanyaan, jangan ragu untuk bertanya!
