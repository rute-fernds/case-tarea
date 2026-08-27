from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import json


def configurar_banco_vetorial(caminho_banco: str) -> StorageContext:
    try:
        Settings.chunk_size = 384 
        Settings.chunk_overlap = 50
        Settings.text_splitter = SentenceSplitter(chunk_size=384, chunk_overlap=50)
        Settings.embed_batch_size = 32 
        Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small", device="cpu", text_instruction="passage: ")
        vetores = LanceDBVectorStore(uri=caminho_banco, table_name="documentos_tarea", mode="overwrite")
        contexto = StorageContext.from_defaults(vector_store=vetores)
        
        return contexto
    except Exception as e:
        print(f"Erro ao configurar LanceDB: {e}")
        return None


def indexar_documentos(caminho_json: str, armazenmaneto_contexto: StorageContext) -> VectorStoreIndex:
    try:
        with open(caminho_json, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        documentos = []
        for doc in dados:
            documentos.append(Document(
                text=doc.get("text", ""),
                metadata=doc.get("metadata", {})
            ))

        print(f"Iniciando vetorização de {len(documentos)} documentos.")
        index = VectorStoreIndex.from_documents(documentos, storage_context=armazenmaneto_contexto)
        print("Vetorização concluída.")
        
        return index
    except Exception as e:
        print(f"Erro ao indexar os documentos: {e}")
        return None