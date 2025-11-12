# 🚀 OCR + RAG System - Panduan Lengkap

Sistem OCR + RAG (Retrieval-Augmented Generation) yang lengkap dan mudah digunakan dengan OpenAI API.

---

## 📁 Struktur File

```
.
├── .env                      # ⚙️  Konfigurasi API key (EDIT INI!)
├── .env.example              # 📄 Template konfigurasi
│
├── MULAI_DISINI.md          # 🚀 START HERE! Panduan quick start
├── PANDUAN_LENGKAP.md       # 📚 Tutorial lengkap & detail
├── README_INDONESIA.md      # 📖 File ini
│
├── test_setup.py            # 🧪 Test konfigurasi sistem
├── quick_start.py           # ⚡ Contoh paling sederhana (5 menit)
├── demo_complete.py         # 🎮 Demo interaktif lengkap
├── contoh_lengkap.py        # 📚 10 contoh siap pakai
│
├── rag_system.py            # 🔧 Core: RAG system
├── ocr_rag_pipeline.py      # 🔧 Core: OCR + RAG pipeline
│
└── requirements.txt         # 📦 Dependencies
```

---

## 🎯 Quick Start (5 Menit)

### 1️⃣ Set API Key

Edit file `.env`:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
```

Dapatkan API key di: https://platform.openai.com/api-keys

### 2️⃣ Test Setup

```bash
python3 test_setup.py
```

### 3️⃣ Run Quick Start

```bash
python3 quick_start.py
```

**Selesai!** 🎉

---

## 📚 File-File Penting

### 🚀 MULAI_DISINI.md
**Baca ini pertama kali!**

Panduan quick start untuk setup dalam 5 menit:
- Setup API key
- Test konfigurasi
- Run contoh pertama

### 📖 PANDUAN_LENGKAP.md
**Tutorial lengkap & detail**

Mencakup:
- Setup detail step-by-step
- Penjelasan setiap komponen
- Use cases praktis
- Troubleshooting
- Tips & tricks

### 🧪 test_setup.py
**Test konfigurasi sistem**

Verifikasi:
- ✅ Dependencies terinstall
- ✅ File .env valid
- ✅ API key benar
- ✅ Koneksi ke OpenAI OK

```bash
python3 test_setup.py
```

### ⚡ quick_start.py
**Contoh paling sederhana**

Contoh minimal untuk mulai cepat:
- Initialize RAG
- Add documents
- Query dan dapat jawaban

```bash
python3 quick_start.py
```

### 🎮 demo_complete.py
**Demo interaktif lengkap**

4 demo interaktif:
1. RAG System Dasar
2. OCR + RAG
3. Dokumen Custom
4. Mode Interaktif (Tanya Jawab)

```bash
python3 demo_complete.py
```

### 📚 contoh_lengkap.py
**10 contoh siap pakai**

Contoh lengkap untuk berbagai use case:
1. RAG paling sederhana
2. Dengan metadata
3. Custom system prompt
4. Batch processing
5. Multi-language
6. OCR + RAG
7. Index dari folder
8. Save & load index
9. Different LLM models
10. Error handling

```bash
python3 contoh_lengkap.py
```

---

## 🔧 Core Files

### rag_system.py
**Core RAG system**

Class: `LightweightRAG`

Fitur:
- FAISS untuk similarity search
- Support multiple LLM providers
- Auto save/load index
- Lightweight (~100MB memory)

**Contoh penggunaan:**

```python
from rag_system import LightweightRAG

rag = LightweightRAG()
rag.add_documents(["Dokumen 1", "Dokumen 2"])
result = rag.query("Pertanyaan?")
print(result['answer'])
```

### ocr_rag_pipeline.py
**OCR + RAG pipeline**

Class: `OCRRAGPipeline`

Fitur:
- OCR sederhana (pytesseract)
- DeepSeek-OCR (hasil terbaik)
- Auto indexing ke RAG
- Batch processing

**Contoh penggunaan:**

```python
from ocr_rag_pipeline import OCRRAGPipeline

pipeline = OCRRAGPipeline(use_gpu=False)
pipeline.index_images(["image1.jpg", "image2.jpg"])
result = pipeline.query("Apa isi dokumen?")
print(result['answer'])
```

---

## 🎓 Learning Path

### Pemula (Hari 1)
1. ✅ Baca **MULAI_DISINI.md**
2. ✅ Run `test_setup.py`
3. ✅ Run `quick_start.py`
4. ✅ Coba **demo_complete.py** (Demo 1 & 4)

### Intermediate (Hari 2-3)
1. ✅ Baca **PANDUAN_LENGKAP.md**
2. ✅ Coba semua contoh di `contoh_lengkap.py`
3. ✅ Modifikasi contoh untuk data Anda
4. ✅ Coba OCR + RAG (Demo 2)

### Advanced (Hari 4+)
1. ✅ Baca source code `rag_system.py`
2. ✅ Customize untuk use case Anda
3. ✅ Integrate ke aplikasi Anda
4. ✅ Deploy ke production

---

## 💡 Use Cases

### 1. Document Q&A
```python
# Index dokumen
rag.add_documents(documents)

# Tanya jawab
result = rag.query("Apa isi kontrak?")
```

### 2. Image OCR + Search
```python
# OCR + index images
pipeline.index_images(image_paths)

# Search dalam gambar
result = pipeline.query("Cari invoice bulan Januari")
```

### 3. Knowledge Base
```python
# Build knowledge base
rag.add_documents(knowledge_articles)

# Query knowledge base
result = rag.query("Bagaimana cara reset password?")
```

### 4. Multi-language Support
```python
# Multilingual embedding
rag = LightweightRAG(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
)

# Add docs dalam berbagai bahasa
rag.add_documents(multilingual_docs)
```

---

## 🔍 Troubleshooting

### Problem: API key error
```bash
# Check .env file
cat .env

# Pastikan format: OPENAI_API_KEY=sk-...
```

### Problem: Module not found
```bash
# Install dependencies
pip install faiss-cpu sentence-transformers openai python-dotenv
```

### Problem: Quota exceeded
- Isi saldo: https://platform.openai.com/account/billing

### Problem: Slow performance
- Gunakan embedding model lebih kecil
- Reduce top_k parameter
- Use GPU jika tersedia

**Lihat PANDUAN_LENGKAP.md untuk troubleshooting detail!**

---

## 📊 Performa

### Memory Usage
- Embedding model: ~80MB
- FAISS index (1000 docs): ~1.5MB
- **Total: ~100MB** (sangat ringan!)

### Biaya OpenAI
- GPT-3.5-turbo: ~$0.002 per 1K tokens
- 1 query: ~$0.001
- 1000 queries: ~$1
- **Sangat murah!**

### Speed
- Embedding: ~1000 docs/second
- FAISS search: <1ms
- LLM generation: 1-3 seconds

---

## 🎯 Next Steps

1. ✅ **Setup:** Ikuti MULAI_DISINI.md
2. ✅ **Learn:** Baca PANDUAN_LENGKAP.md
3. ✅ **Practice:** Coba semua contoh
4. ✅ **Build:** Buat aplikasi Anda!

---

## 📞 Support

- **Documentation:** PANDUAN_LENGKAP.md
- **Examples:** contoh_lengkap.py
- **Test:** test_setup.py
- **OpenAI Docs:** https://platform.openai.com/docs

---

## 📄 License

MIT License - Bebas digunakan untuk project apapun!

---

## 🎉 Selamat Mencoba!

Mulai dari **MULAI_DISINI.md** dan ikuti step-by-step.

Dalam 5 menit Anda sudah bisa running sistem RAG pertama Anda! 🚀

---

**Made with ❤️ for Indonesian developers**
