<h1 align="center">LLM From Scratch</h1>

<p align="center">
  <b>Mini-GPT Style Transformer | PyTorch | Custom Tokenizer | FastAPI Web UI</b>
</p>

<p align="center">
  <b>Developed by:</b> <a href="#">M. Sowmyapriya</a> & <a href="#">Lalitha</a>
</p>

<p align="center">
  <a href="#project-overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#impact">Impact</a>
</p>

---

## 🚀 Project Overview
**LLM From Scratch** is a **miniature GPT-style language model** built entirely from scratch in **PyTorch**.  
It is designed for **learning, experimentation, and research**, showcasing a full **tokenization → training → inference pipeline**.  

This project demonstrates the capability to design **custom language models**, implement **transformer architectures**, and integrate them with a **FastAPI backend** and **browser-based web interface**.

> ⚠️ **Note:** Large checkpoints (~4GB) are excluded from this repo. Download separately from [Google Drive](#) and place in `checkpoints/`.

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **Custom Tokenizer** | BPE tokenizer with optional HuggingFace integration |
| **Transformer Architecture** | Multi-head causal attention, feed-forward layers, layer normalization |
| **Text Generation** | Top-k and nucleus (top-p) sampling for high-quality outputs |
| **Training Pipeline** | Mixed precision, checkpointing, learning rate scheduling |
| **Web UI** | Browser-based interactive chat with FastAPI backend |
| **Lightweight & Modular** | Clear project structure for experimentation and learning |

---

## 🗂 Repository Structure

**llm_project/** – Root folder for the LLM project  
├─ **README.md** – Project overview and instructions  
├─ **config.yaml** – Training and model configuration parameters  
├─ **requirements.txt** – Python dependencies  
├─ **run_llm.py** – Script to run training or inference  
├─ **data/** – Sample datasets (e.g., `tiny_shakespeare.txt`)  
├─ **training/** – Training scripts, checkpoints, and utilities  
├─ **model/** – Transformer model implementation (attention, feed-forward layers, etc.)  
├─ **tokenizer/** – Tokenizer scripts and vocabulary files  
├─ **inference/** – Scripts for generating text using the trained model  
├─ **notebooks/** – Experiment notes and analysis in Markdown/Jupyter notebooks  
└─ **web_ui/** – Browser-based web interface (FastAPI backend + HTML/JS frontend)  

> ⚠️ **Note:** Large model checkpoints (~4+ GB) are excluded from this repo. Download separately from [Google Drive](#) and place in `checkpoints/` before running inference.


---

## 🎯 Features

- **Tokenizer:** Custom BPE, HuggingFace-compatible  
- **Transformer Model:** Multi-head attention, feed-forward layers, layer normalization  
- **Text Generation:** Supports top-k and top-p sampling  
- **Training:** Mixed precision, learning rate scheduling, checkpointing  
- **Web Interface:** Interactive browser-based chat connected to FastAPI  

---

## 🛠 Tech Stack

**Languages:** Python, JavaScript  
**AI/ML:** PyTorch, Transformers, Tokenizers, Whisper  
**Web/API:** FastAPI, HTML/JS  
**Tools:** Jupyter Notebook, Git, GitHub  
**Data Storage:** Local datasets + external checkpoints  

---

## 🚀 Quick Start

# 1. Create & activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset
# Place training file at data/tiny_shakespeare.txt

# 4. Train the model
python training/trainer.py --config config.yaml

# 5. Start FastAPI server
python api/server.py

# 6. Launch web UI
# Open web_ui/static/index.html in your browser

💡 Why This Project Matters

Demonstrates end-to-end LLM pipeline from scratch
Provides hands-on experience with Transformer internals
Ideal for learning, research, and portfolio showcase

🌐 Connect With Authors

M. Sowmyapriya: Coming Soon| sowmyapriya7325@gmail.com

Lalitha: Coming Soon | lalitha.koruprolu29@gmail.com
