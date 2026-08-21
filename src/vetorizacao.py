import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

Settings.chunk_size = 512
Settings.chunk_overlap = 50

def processar_vetorizacao(documentos: list, caminho_db: str = "./chroma_db") -> VectorStoreIndex:
    """
    Fatia (Chunks), vetoriza e armazena documentos em um banco de dados vetorial, o ChromaDB.

    Args:
        documentos (list): Lista de objetos Document gerados pela etapa de extração e classificação.
        caminho_db (str, optional): Caminho do diretório onde o banco físico será criado ou atualizado.

    Returns:
        VectorStoreIndex: O índice de orquestração construído sobre os dados vetorizados.
    """
    if not documentos:
        print("Nenhum documento para vetorização.")
        return None

    print(f"Vetorizando {len(documentos)} páginas.")

    try:
        db = chromadb.PersistentClient(path=caminho_db)
        
        colecao_chroma = db.get_or_create_collection("documentos_tarea")
        
        vetor_store = ChromaVectorStore(chroma_collection=colecao_chroma)
        storage_context = StorageContext.from_defaults(vector_store=vetor_store)
        
        indice = VectorStoreIndex.from_documents(
            documentos, 
            storage_context=storage_context,
            show_progress=True
        )
        
        print(f"Banco vetorial criado na pasta: '{caminho_db}'!")
        return indice

    except Exception as erro:
        print(f"Erro durante a vetorização: {erro}")
        return None