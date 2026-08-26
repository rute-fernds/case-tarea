from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import warnings
from config import CAMINHO_DB

warnings.filterwarnings("ignore")

def iniciar_chat(caminho_banco: str = "./lancedb"):
    print("Carregando modelos...")

    modelo_llm = Ollama(model="gemma2:2b", request_timeout=180.0)
    Settings.llm = modelo_llm

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

    vector_store = LanceDBVectorStore(uri=caminho_banco, query_type="hybrid")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    retriever = index.as_retriever(
        similarity_top_k=3,
        vector_store_query_mode="hybrid"
    )

    prompt_contexto = (
    "Você é um analista jurídico virtual brasileiro. Responda de forma direta e formal, "
    "seguindo RIGOROSAMENTE as regras abaixo:\n\n"
    "1. IDIOMA: Responda SEMPRE em Português do Brasil.\n"
    "2. CONTEXTO EXCLUSIVO: Baseie-se APENAS nos 'Documentos de Contexto'. Nunca invente. "
    "Se a resposta não constar no texto, diga exatamente: 'Não encontrei essa informação específica nos documentos fornecidos.'\n"
    "3. REGRAS E EXCEÇÕES: Diferencie claramente as diretrizes gerais de suas respectivas exceções.\n"
    "4. DADOS CRÍTICOS: Extraia com exatidão prazos, termos técnicos e exigências profissionais.\n"
    "5. DESCUMPRIMENTO: Se questionado sobre falhas ou omissões, foque estritamente nas sanções e penalidades descritas no texto.\n"
    "Documentos de Contexto:\n"
    "{context_str}")

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=modelo_llm, 
        context_prompt=prompt_contexto,
        verbose=False
    )

    print("RAG Híbrido")
    print("Digite 'sair' para encerrar ou 'fontes' após uma resposta para ver de onde a IA tirou a informação.")
    print("="*50 + "\n")

    ultima_resposta = None

    while True:
        pergunta = input("\nUsuário: ")
        
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando o chat...")
            break
            
        if pergunta.lower() == 'fontes':
            if ultima_resposta and ultima_resposta.source_nodes:
                print("\n--- Fontes Utilizadas ---")
                for node in ultima_resposta.source_nodes:
                    meta = node.node.metadata
                    print(f"Arquivo: {meta.get('name_file')} (Pág {meta.get('page_label')}) | Score: {node.score:.4f}")
            else:
                print("Nenhuma fonte disponível para a última pergunta.")
            continue

        print("\nBuscando documentos...")
        
        try:
            ultima_resposta = chat_engine.chat(pergunta)
            print(f"\n🤖 Gemma2.2b: {ultima_resposta.response}")
        except Exception as e:
            print(f"\nErro de comunicação com o LLM: {e}")


if __name__ == "__main__":
    iniciar_chat(CAMINHO_DB)