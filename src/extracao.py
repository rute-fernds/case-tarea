from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
import re
import unicodedata

leitor_pdf = PyMuPDFReader()
arquivo_tipo = {".pdf": leitor_pdf}


def extrair_texto_pdf(diretorio: str) -> list:
    """
    Extrai o texto bruto de todos os arquivos PDF de uma pasta específica.

    Args:
        diretorio (str): O caminho da pasta onde os PDFs estão salvos.

    Returns:
        list: Uma lista de objetos do tipo Document contendo o texto e metadados.
    """

    try:
        documentos_extraidos = SimpleDirectoryReader(
            input_dir=diretorio,
            file_extractor=arquivo_tipo,
            required_exts=[".pdf"]
        ).load_data()

        return documentos_extraidos
    
    except Exception as erro:
        print(f"Erro ao extrair documentos do diretório '{diretorio}': {erro}")
        return []


def limpar_texto(texto: str) -> str:
    """Remove quebras de linha irregulares, hifenização e normaliza os caracteres."""

    if not texto:
        return ""
    try:
        texto_limpo = unicodedata.normalize('NFKC', texto)
        texto_limpo = re.sub(r'-\n', '', texto_limpo)
        texto_limpo = re.sub(r'(?<!\n)\n(?!\n)', ' ', texto_limpo)
        texto_limpo = re.sub(r'\s+', ' ', texto_limpo)

        return texto_limpo.strip()

    except Exception as erro:
        print(f"Erro na normalização de texto: {erro}")
        return texto


def processar_extracao(diretorio: str) -> list:
    """
    Extrai o texto dos PDFs, remove páginas vazias
    e aplica a normalização no conteúdo de cada documento.
    """

    try:
        print(f"Iniciando o pipeline de extração e normalização de PDFs do diretório: {diretorio}")
        documentos_extraidos = extrair_texto_pdf(diretorio)
        documentos_processados = []

        for documento in documentos_extraidos:
            if documento.text.strip():
                texto_normalizado = limpar_texto(documento.text)
                documento.set_content(texto_normalizado)
                documentos_processados.append(documento)

        print(f"{len(documentos_processados)} páginas processadas.")
        return documentos_processados

    except Exception as erro:
        print(f"Erro crítico no pipeline da Fase 1: {erro}")
        return []
