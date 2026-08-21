import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("Conectando ao LLM Gemma 2 2B...")

Settings.llm = Ollama(model="gemma2:2b", request_timeout=120.0)


def iniciar_chat(caminho_db: str = "./chroma_db"):
    print("\nConectando ao ChromaDB...")
    try:
        db = chromadb.PersistentClient(path=caminho_db)
        colecao_chroma = db.get_collection("documentos_tarea")
        
        vetor_store = ChromaVectorStore(chroma_collection=colecao_chroma)
        storage_context = StorageContext.from_defaults(vector_store=vetor_store)
        
        indice = VectorStoreIndex.from_vector_store(
            vector_store=vetor_store,
            storage_context=storage_context
        )
    
        prompt_sistema = (
            "Você é um assistente jurídico virtual especialista em leis brasileiras. "
            "Sempre responda em português do Brasil de forma clara, direta e profissional. "
            "Use APENAS o contexto fornecido para responder."
        )
        
        query_engine = indice.as_query_engine(
            similarity_top_k=6,
            system_prompt=prompt_sistema
        )

        print("="*50)
        
        while True:
            pergunta = input("\n😭 Usuário: ")
            if pergunta.lower() in ['sair', 'exit', 'quit']:
                print("Assitente encerrado.")
                break
                
            print("🤖 Gemma:")
            resposta = query_engine.query(pergunta)
            
            print(f"\n> {resposta}\n")
            print("📌 Fontes utilizadas:")
            for node in resposta.source_nodes:
                nome = node.metadata.get("file_name", "Desconhecido")
                print(f" - 📄 {nome}")
                
    except Exception as erro:
        print(f"Erro ao iniciar o chat: {erro}")

if __name__ == "__main__":
    iniciar_chat()