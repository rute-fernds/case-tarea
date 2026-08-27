import warnings
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import LongContextReorder
from config import CAMINHO_DB

warnings.filterwarnings("ignore")

def iniciar_chat(caminho_banco: str) -> CondensePlusContextChatEngine:
    modelo_llm = Ollama(model="gemma2:2b", request_timeout=180.0, temperature=0.0)
    Settings.llm = modelo_llm

    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small", device="cpu", query_instruction="query: ")

    vector_store = LanceDBVectorStore(uri=caminho_banco, table_name="documentos_tarea")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    retriever = index.as_retriever(similarity_top_k=7, vector_store_query_mode="default")

    prompt_contexto = (
        "Você é um assistente virtual brasileiro especialista em extração literal de dados de documentos normativos.\n"
        "Sua tarefa é responder à pergunta do usuário de forma clara, direta e formal, seguindo RIGOROSAMENTE as diretrizes abaixo:\n\n"
        "1. ANCORAGEM LITERAL: Responda baseando-se ÚNICA E EXCLUSIVAMENTE nas informações explícitas dos 'Documentos de Contexto' abaixo. Nunca invente, deduza ou use conhecimentos externos (como idades, sistemas de ensino de outros países ou jargões genéricos).\n"
        "2. ADMISSÃO DE AUSÊNCIA: Se a resposta ou os detalhes técnicos exatos (prazos, valores, exceções) não estiverem explicitamente escritos no texto, responda EXATAMENTE: 'Não encontrei essa informação nos documentos fornecidos.' Nunca tente adivinhar ou preencher lacunas com suposições.\n"
        "3. PRESERVAÇÃO DE ESCOPOS: Preste atenção para não misturar as regras de entidades distintas. Se a pergunta for sobre municípios, não responda com regras de estados. Se for sobre bicicletas elétricas, não aplique regras de ciclomotores.\n"
        "4. TERMINOLOGIA OFICIAL: Utilize estritamente os termos técnicos do direito brasileiro presentes no texto (ex: use 'restituídos' ou 'devolvidos' em vez de traduções livres como 'restaurados').\n"
        "5. FATOS: Limite-se aos fatos diretos do texto. É proibido criar listas de competências comportamentais, éticas ou de liderança que não estejam explicitamente escritas na norma.\n\n"
        "6. BASE EXCLUSIVA: Responda baseando-se ÚNICA E EXCLUSIVAMENTE nos 'Documentos de Contexto' abaixo."
        "Documentos de Contexto:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
    )

    reorder = LongContextReorder()

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        node_postprocessors=[reorder],
        llm=modelo_llm, 
        context_prompt=prompt_contexto,
        verbose=False
    )
    
    return chat_engine


if __name__ == "__main__":
    chat_gemma = iniciar_chat(CAMINHO_DB)
    
    print("Chat com Gemma2:2b iniciado")
    print("="*50)

    ultima_resposta = None

    while True:
        pergunta = input("\n😄 Usuário: ")
        
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando Gemma2:2b...")
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

        print("\nConsultando documentos...")
        
        try:
            ultima_resposta = chat_gemma.chat(pergunta)
            print(f"\n🤖 Gemma2:2b: {ultima_resposta.response}")
        except Exception as e:
            print(f"\nErro de comunicação com o LLM: {e}")