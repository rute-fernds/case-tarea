import os
from config import CAMINHO_PDFS, CAMINHO_DB, CAMINHO_JSON_CLASSIFICADO, CAMINHO_JSON_EXTRAIDO
from extracao import extrair_texto_pdf
from classificacao import processar_classificacao
from vetorizacao import configurar_banco_vetorial, indexar_documentos


def inicializar_rag() -> bool:
    try:
        extrair_texto_pdf(CAMINHO_PDFS, CAMINHO_JSON_EXTRAIDO) 
        processar_classificacao(CAMINHO_JSON_EXTRAIDO, CAMINHO_JSON_CLASSIFICADO)
        
        contexto = configurar_banco_vetorial(CAMINHO_DB)
        if contexto:
            indexar_documentos(CAMINHO_JSON_CLASSIFICADO, contexto)
        else:
            raise Exception("Falha ao configurar o contexto de armazenamento do banco vetorial.")

        print("\nPipeline concluída.")
        return True
    
    except Exception as e:
        print(f"Erro ao inicializar sistema: {e}")
        return False

if __name__ == "__main__":
    inicializar_rag()