# RAG System untuk DeepSeek-OCR

Sistem RAG (Retrieval-Augmented Generation) ringan dengan FAISS untuk DeepSeek-OCR.

## 🚀 Fitur

- ✅ **Ringan**: Menggunakan FAISS (CPU) dan embedding model kecil (~80MB)
- ✅ **Multi-LLM**: Support OpenAI, DeepSeek, Groq, dan Ollama (local)
- ✅ **Flexible**: Bisa pakai DeepSeek-OCR atau OCR sederhana (pytesseract)
- ✅ **Fast**: FAISS untuk similarity search yang cepat

## 📦 Instalasi

### 1. Install Dependencies

```bash
pip install -r rag_requirements.txt
```

### 2. Setup API Key

Copy `.env.example` ke `.env`:

```bash
cp .env.example .env
```

Edit `.env` dan pilih salah satu provider:

#### Option 1: OpenAI (Berbayar)
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
```

**Cara dapat API key:**
1. Kunjungi: https://platform.openai.com/api-keys
2. Login/daftar
3. Klik "Create new secret key"
4. Copy dan paste ke `.env`
5. Isi saldo di: https://platform.openai.com/account/billing

#### Option 2: DeepSeek (Murah, Recommended)
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
```

**Cara dapat API key:**
1. Kunjungi: https://platform.deepseek.com/
2. Daftar akun
3. Buat API key
4. Harga sangat murah: ~$0.14 per 1M tokens

#### Option 3: Groq (Gratis Tier)
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
LLM_PROVIDER=groq
LLM_MODEL=mixtral-8x7b-32768
```

**Cara dapat API key:**
1. Kunjungi: https://console.groq.com/
2. Daftar akun (gratis)
3. Buat API key
4. Free tier: 14,400 requests/day

#### Option 4: Ollama (Local, Gratis)
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

**Cara install Ollama:**
1. Download: https://ollama.ai/download
2. Install dan jalankan
3. Pull model: `ollama pull llama2`
4. Tidak perlu API key!

## 🎯 Penggunaan

### 1. RAG System Dasar

```python
from rag_system import LightweightRAG

# Initialize
rag = LightweightRAG(
    llm_provider="openai",  # atau "deepseek", "groq", "ollama"
    llm_model="gpt-3.5-turbo"
)

# Add documents
documents = [
    "DeepSeek-OCR adalah model OCR yang powerful.",
    "Model ini support berbagai resolusi."
]
rag.add_documents(documents)

# Query
result = rag.query("Apa itu DeepSeek-OCR?")
print(result['answer'])
```

### 2. Pipeline Lengkap (OCR + RAG)

```python
from ocr_rag_pipeline import OCRRAGPipeline

# Initialize pipeline
pipeline = OCRRAGPipeline(
    use_gpu=False,  # Set True jika punya GPU
    rag_config={
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo"
    }
)

# Index images
image_paths = ["image1.jpg", "image2.jpg"]
pipeline.index_images(
    image_paths,
    use_deepseek_ocr=False  # Set True untuk hasil terbaik (butuh GPU)
)

# Query
result = pipeline.query("Apa isi dokumen?")
print(result['answer'])
```

### 3. Demo Script

```bash
# Test RAG system
python rag_system.py

# Test full pipeline
python ocr_rag_pipeline.py
```

## 🔧 Konfigurasi

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2` (ringan, ~80MB)

Alternatif:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)
- `sentence-transformers/all-mpnet-base-v2` (lebih akurat, tapi lebih besar)

### OCR Mode

**Simple OCR (Ringan):**
```python
pipeline.index_images(image_paths, use_deepseek_ocr=False)
```
- Menggunakan pytesseract
- Tidak butuh GPU
- Install: `pip install pytesseract`

**DeepSeek-OCR (Terbaik):**
```python
pipeline.index_images(image_paths, use_deepseek_ocr=True)
```
- Hasil OCR terbaik
- Butuh GPU (recommended)
- Model besar (~10GB)

## 💡 Tips

### 1. Pilih LLM Provider

| Provider | Kelebihan | Kekurangan |
|----------|-----------|------------|
| **OpenAI** | Paling populer, hasil bagus | Berbayar, agak mahal |
| **DeepSeek** | Murah, cocok untuk OCR | Perlu daftar |
| **Groq** | Gratis tier, cepat | Rate limit |
| **Ollama** | Gratis, private | Butuh install local |

**Rekomendasi:**
- **Untuk production**: DeepSeek (murah) atau OpenAI (reliable)
- **Untuk testing**: Groq (gratis) atau Ollama (local)

### 2. Hemat Resource

```python
# Gunakan embedding model kecil
rag = LightweightRAG(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Gunakan simple OCR jika tidak punya GPU
pipeline.index_images(images, use_deepseek_ocr=False)

# Limit jumlah retrieved documents
result = rag.query(question, top_k=3)
```

### 3. Untuk Dataset Besar

```python
# Process images in batches
batch_size = 10
for i in range(0, len(all_images), batch_size):
    batch = all_images[i:i+batch_size]
    pipeline.index_images(batch)
```

## 📊 Performa

**Embedding Model (all-MiniLM-L6-v2):**
- Size: ~80MB
- Speed: ~1000 docs/second
- Memory: ~200MB

**FAISS Index:**
- Search: O(n) untuk IndexFlatL2
- Memory: ~4 bytes per dimension per document
- Untuk 1000 docs (384 dim): ~1.5MB

**Total Memory (tanpa OCR):**
- ~300MB untuk RAG system
- Sangat ringan!

## 🐛 Troubleshooting

### Error: API Key tidak ditemukan
```bash
# Pastikan .env file ada dan berisi API key
cat .env

# Atau set manual
export OPENAI_API_KEY=sk-xxxxx
```

### Error: Model terlalu besar
```python
# Gunakan simple OCR
pipeline = OCRRAGPipeline(use_gpu=False)
pipeline.index_images(images, use_deepseek_ocr=False)
```

### Error: CUDA out of memory
```python
# Gunakan CPU mode
pipeline = OCRRAGPipeline(use_gpu=False)
```

### Error: pytesseract not found
```bash
# Install pytesseract
pip install pytesseract

# Linux: install tesseract
sudo apt-get install tesseract-ocr

# Mac: 
brew install tesseract
```

## 📚 Contoh Use Cases

### 1. Index Dokumen dari Folder

```python
from pathlib import Path

# Get all images
images = list(Path("./documents").glob("*.jpg"))

# Index
pipeline.index_images([str(img) for img in images])

# Query
result = pipeline.query("Cari informasi tentang invoice")
```

### 2. Multi-language Support

```python
# Gunakan multilingual embedding
rag = LightweightRAG(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Add documents dalam berbagai bahasa
docs = [
    "DeepSeek-OCR is a powerful OCR model",
    "DeepSeek-OCR adalah model OCR yang powerful"
]
rag.add_documents(docs)
```

### 3. Custom System Prompt

```python
custom_prompt = """Anda adalah asisten untuk analisis dokumen keuangan.
Fokus pada angka, tanggal, dan informasi penting."""

result = rag.query(
    "Berapa total invoice?",
    system_prompt=custom_prompt
)
```

## 🎓 Next Steps

1. **Improve OCR**: Fine-tune DeepSeek-OCR untuk domain spesifik
2. **Better Chunking**: Split dokumen panjang menjadi chunks
3. **Metadata Filtering**: Filter berdasarkan tanggal, kategori, dll
4. **Hybrid Search**: Combine FAISS dengan keyword search
5. **UI**: Buat web interface dengan Streamlit/Gradio

## 📞 Support

- **DeepSeek-OCR**: https://github.com/deepseek-ai/DeepSeek-OCR
- **FAISS**: https://github.com/facebookresearch/faiss
- **LangChain**: https://python.langchain.com/

## 📄 License

MIT License - lihat LICENSE file
