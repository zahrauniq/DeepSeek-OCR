# 🚀 MULAI DISINI - Quick Setup Guide

Panduan cepat untuk memulai dalam **5 menit**!

---

## ✅ Step 1: Set API Key OpenAI (WAJIB!)

### 1. Dapatkan API Key

Kunjungi: **https://platform.openai.com/api-keys**

- Login atau daftar
- Klik **"Create new secret key"**
- Copy API key (format: `sk-proj-...`)

### 2. Edit File .env

Buka file `.env` dan ganti:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Dengan API key Anda:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
```

**PENTING:** Simpan file setelah edit!

---

## ✅ Step 2: Test Setup

Jalankan test untuk memastikan semuanya OK:

```bash
python3 test_setup.py
```

**Output yang diharapkan:**

```
✅ Imports
✅ Environment File
✅ RAG System
✅ OCR Pipeline
✅ API Connection

🎉 Semua test passed! Sistem siap digunakan!
```

---

## ✅ Step 3: Quick Start

Jalankan contoh sederhana:

```bash
python3 quick_start.py
```

Ini akan:
1. ✅ Load API key
2. ✅ Initialize RAG system
3. ✅ Add sample documents
4. ✅ Query dan dapat jawaban

---

## ✅ Step 4: Demo Lengkap (Opsional)

Untuk eksplorasi lebih lanjut:

```bash
python3 demo_complete.py
```

Menu demo:
- **Demo 1:** RAG System Dasar
- **Demo 2:** OCR + RAG
- **Demo 3:** Dokumen Custom
- **Demo 4:** Mode Interaktif (Tanya Jawab)

---

## 📚 Dokumentasi Lengkap

Baca **PANDUAN_LENGKAP.md** untuk:
- Tutorial detail
- Contoh code
- Use cases praktis
- Troubleshooting

---

## 🐛 Troubleshooting Cepat

### Problem: API key error

```bash
# Check isi file .env
cat .env

# Pastikan format benar:
# OPENAI_API_KEY=sk-proj-...
# (tidak ada spasi, tidak ada quotes)
```

### Problem: Module not found

```bash
# Install ulang dependencies
pip install faiss-cpu sentence-transformers openai python-dotenv tiktoken
```

### Problem: Quota exceeded

- Isi saldo di: https://platform.openai.com/account/billing
- Minimum: $5

---

## 💻 Contoh Code Minimal

```python
from dotenv import load_dotenv
from rag_system import LightweightRAG

load_dotenv()

# Initialize
rag = LightweightRAG(
    llm_provider="openai",
    llm_model="gpt-3.5-turbo"
)

# Add documents
rag.add_documents([
    "Dokumen pertama...",
    "Dokumen kedua..."
])

# Query
result = rag.query("Pertanyaan saya?")
print(result['answer'])
```

---

## 📞 Butuh Bantuan?

1. **Baca:** PANDUAN_LENGKAP.md
2. **Test:** python3 test_setup.py
3. **Demo:** python3 demo_complete.py

---

**Selamat mencoba! 🎉**

Jika sudah berhasil, lanjut ke PANDUAN_LENGKAP.md untuk tutorial lengkap!
