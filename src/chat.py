import warnings
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import LongContextReorder
from src.config import CAMINHO_DB

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
    "Você é um extrator de dados literal de documentos jurídicos brasileiros.\n"
    "Responda à pergunta de forma direta, formal e estritamente ancorada no texto, seguindo estas regras:\n\n"
    
    "1. ANCORAGEM LITERAL: Use apenas o que está explícito no contexto. Nunca use conhecimento externo, jargões genéricos ou invente conceitos (ex: é proibido criar divisões como 'Ensino Médio I e II' ou inventar obrigações).\n"
    "2. ADMISSÃO DE AUSÊNCIA: Se o dado exato (prazos, valores, itens de uma lista) não estiver escrito, responda apenas: 'Não encontrei essa informação nos documentos fornecidos.' Nunca adivinhe ou complete lacunas.\n"
    "3. RETIFICAÇÕES: Em textos de retificação, use APENAS a redação do campo 'Leia-se' ou 'Nova Redação'. Ignore o texto antigo do campo 'Onde se lê'.\n"
    "4. PRESERVAÇÃO DE ESCOPOS: Mantenha distinções rígidas entre entidades (ex: regras de 'Estado' não se aplicam a 'Município'; 'Impedimento' não se mistura com 'Suspeição').\n"
    "5. TERMOS E SIGLAS: Use termos técnicos exatos do texto (ex: se o texto diz 'Educação Infantil', não mude para 'Pré-escolar'). Se o significado de uma sigla (como Cogepi) não estiver escrito por extenso, use apenas a sigla; nunca invente seu significado.\n\n"
    "6. PROIBIÇÃO DE EXPANSÃO: Ao responder sobre um termo, infração ou conceito, limite-se estritamente ao que o texto diz que ele é. É terminantemente proibido explicar o significado de termos com suas próprias palavras ou listar exemplos, consequências e punições que não estejam escritos de forma literal no documento (ex: se o texto diz apenas 'falta grave', nunca liste punições como 'demissão' ou 'suspensão')."
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
    
    print("Gemma - Assistente de Documentos")
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

        print("\nGerando resposta...")
        
        try:
            ultima_resposta = chat_gemma.chat(pergunta)
            print(f"\n🤖 Gemma2:2b: {ultima_resposta.response}")
        except Exception as e:
            print(f"\nErro de comunicação com o LLM: {e}")