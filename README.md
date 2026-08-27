## ✨ Gemma IA: Assistente de Documentos | Case Técnico Tarea
Este Estudo de Caso tem como objetivo o desenvolvimento de um sistema de RAG (Retrieval-Augmented Generation), capaz de consultar e utilizar informações extraídas de documentos PDF para responder perguntas em linguagem natural. 

---

## 📌 Sobre o Projeto

O sistema foi desenvolvido como parte do case técnico do processo seletivo para estágio da empresa Tarea, com o intuito de desenvolver uma solução capaz de transformar dados não estruturados em uma base consultável por um modelo de LLM (Large Language Model).


## 💻 Tecnologias Utilizadas

- Python 3.12
- LlamaIndex
- Gemma 2:2b (via Ollama)
- intfloat/multilingual-e5-small
- MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
- LanceDB
- Tesseract OCR
- PyMuPDF
- Streamlit

---

## 📁 Estrutura do Projeto

```
📁 case-tecnico-tarea/
├── 📁data/
|    ├── 📁lancedb/
|    ├── 📁processed/
|    |    ├── texto_extraido.json
|    |    └── texto_classificado.json
|    └── 📁raw/
|         ├── Lei_9394_20121996.pdf
|         ├── Lei_9784_29011999.pdf
|         ├── Lei_14945_31072024.pdf
|         ├── Portaria_44_28052025.pdf
|         ├── Portaria_81_25082015.pdf
|         ├── Portaria_2140_04072025.pdf
|         ├── Resolucao_99_156202023.pdf
|         ├── Resolucao_909_28032022.pdf
|         ├── Resolucao_2430_21052025.pdf
|         └── Resolucao_2434_03072025.pdf
|    
├── 📁src/
|    ├── 📁img/
|    |    ├── ia.png
|    |    ├── icone.png
|    |    └── user.png
|    |
|    ├── __init__.py
|    ├── chat.py
|    ├── classificacao.py
|    ├── config.py
|    ├── extracao.py
|    └── vetorizacao.py
|
├── app.py
├── pipeline.py
├── README.md
├── .gitignore
└── requirements.txt
```

## 🚀 Como Rodar o Gemma IA

Siga este passo a passo para configurar o ambiente e executar o sistema na sua máquina.

### 1️⃣ Pré-requisitos
Antes de começar, certifique-se de ter instalado:
* [Python 3.12+](https://www.python.org/downloads/)

---



streamlit run main.py
```
