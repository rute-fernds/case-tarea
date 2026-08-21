from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
import numpy as np
import easyocr
import unicodedata
import fitz  
import re

leitor_ocr = easyocr.Reader(['pt'], gpu=False, verbose=False)
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
        documentos = SimpleDirectoryReader(
            input_dir=diretorio,
            file_extractor=arquivo_tipo,
            required_exts=[".pdf"]
        ).load_data()
        return documentos
    except Exception as erro:
        print(f"Erro ao extrair documentos: {erro}")
        return []


def limpar_texto(texto: str) -> str:
    """Remove quebras de linha irregulares, hifenização, símbolos e normaliza o texto."""
    if not texto:
        return ""
    try:
        texto_limpo = re.sub(r'§', 'parágrafo ', texto)
        texto_limpo = unicodedata.normalize('NFKC', texto_limpo)
        texto_limpo = re.sub(r'(\d+)[oO°º](?!\w)', r'\1º', texto_limpo)
        texto_limpo = re.sub(r'(\d+)[aAª](?!\w)', r'\1ª', texto_limpo)
        texto_limpo = re.sub(r'(?i)\s*(p/|/p|<p>|</p>|\\p)\s*', ' ', texto_limpo)
        texto_limpo = re.sub(r'-\n\s*', '', texto_limpo)
        texto_limpo = re.sub(r'(?<!\n)\n(?!\n)', ' ', texto_limpo)
        texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
        return texto_limpo.strip()

    except Exception as erro:
        print(f"Erro na normalização: {erro}")
        return texto


def processar_ocr(caminho_arquivo: str, numero_pagina: int) -> str:
    """
    Extrai o texto de uma página específica de um arquivo PDF utilizando OCR (EasyOCR).

    Args:
        caminho_arquivo (str): O caminho do arquivo PDF.
        numero_pagina (int): O número da página a ser lida.

    Returns:
        str: Uma string contendo os parágrafos de texto extraídos da página.
    """
    try:
        doc = fitz.open(caminho_arquivo)
        pagina = doc.load_page(numero_pagina - 1)
        
        pix = pagina.get_pixmap(dpi=200) # dpi: pontos por polegada
        
        img_matriz= np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        resultado = leitor_ocr.readtext(img_matriz, detail=0, paragraph=True)
        
        return "\n".join(resultado)
        
    except Exception as erro:
        print(f"Erro no processamento do OCR: {erro}")
        return ""


def processar_extracao(diretorio: str) -> list:
    print(f"Extração e normalização de PDFs da pasta: {diretorio}")
    documentos_extraidos = extrair_texto_pdf(diretorio)
    documentos_processados = []

    contador_paginas = {}
    
    for documento in documentos_extraidos:
        texto_bruto = documento.text.strip()
        arquivo_caminho = documento.metadata.get("file_path", "Desconhecido")

        if arquivo_caminho not in contador_paginas:
            contador_paginas[arquivo_caminho] = 1
        else:
            contador_paginas[arquivo_caminho] += 1

        num_pagina = contador_paginas[arquivo_caminho]
        documento.metadata["page_label"] = str(num_pagina)

        if not texto_bruto and arquivo_caminho != "Desconhecido":
            texto_bruto = processar_ocr(arquivo_caminho, num_pagina)

        if texto_bruto.strip():
            texto_normalizado = limpar_texto(texto_bruto)
            documento.set_content(texto_normalizado)
            documentos_processados.append(documento)
            
    return documentos_processados
