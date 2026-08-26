from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
from config import CAMINHO_JSON_CLASSIFICADO, CAMINHO_DB


def criar_db(caminho_json: str, caminho_banco: str):
    embed_modelo = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

    vetores = LanceDBVectorStore(
        uri=caminho_banco, 
        mode="overwrite", 
        query_type="hybrid")

    storage_context = StorageContext.from_defaults(vector_store=vetores)

    with open(caminho_json, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    documentos = []
    for doc in dados:
        documentos.append(Document(
            text=doc.get("text", ""),
            metadata=doc.get("metadata", {})
        ))

    index = VectorStoreIndex.from_documents(
        documentos, 
        storage_context=storage_context,
        embed_model=embed_modelo
    )
    
    print("Vetorização realizada.")
    return index


if __name__ == "__main__":
    criar_db(CAMINHO_JSON_CLASSIFICADO, CAMINHO_DB)