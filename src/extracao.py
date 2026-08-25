from config import CAMINHO_TESSERACT, CONFIG_TESSERACT
from pathlib import Path
import numpy as np
import pymupdf
import pytesseract
import json
import cv2
import re
import os

pytesseract.pytesseract.tesseract_cmd = CAMINHO_TESSERACT + r"\tesseract.exe"

def pre_processar_img(img_array: np.ndarray) -> np.ndarray:
    try:
        img_cinza = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_gaus = cv2.GaussianBlur(img_cinza, (3, 3), 0)
        _, img_binarizada = cv2.threshold(img_gaus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_binarizada
    except Exception as e:
        print(f"Erro ao tentar realizar o pré-processamento da imagem: {e}") 
        return img_array


def ocr_img(img_processada: np.ndarray) -> str:
    try:
        texto_extraido = pytesseract.image_to_string(img_processada, config=CONFIG_TESSERACT)
        return texto_extraido.strip()
    except Exception as e:
        print(f"Erro ao tentar extrair texto da imagem: {e}")
        return ""


def limpar_texto(texto: str, is_ocr: bool = False) -> str:
    try:
        if not texto:
            return ""
        
        if is_ocr:
            texto = re.sub(r'\b8\s+(?=\d+([º°ª]|\.|-))', '§ ', texto)
            texto = re.sub(r'(?<=[a-zA-Z])\(', ' (', texto)        

        texto = re.sub(r'\n{2,}', '<PARAGRAFO>', texto)
        texto = re.sub(r'\n', ' ', texto)
        texto = texto.replace('<PARAGRAFO>', '\n\n')
        texto_limpo = re.sub(r'[ \t]{2,}', ' ', texto)
        return texto_limpo.strip()

    except Exception as e:
        print(f'Erro ao realizar limpeza no texto extraído: {e}')
        return texto


def salvar_json(documentos: list, caminho_saida_json: str) -> bool:
    try:
        caminho_arquivo = Path(caminho_saida_json)
        caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            json.dump(documentos, f, ensure_ascii=False, indent=4)
        print(f"JSON salvo em: {caminho_arquivo}")
        return True
    except Exception as e:
        print(f"Erro ao salvar arquivo JSON: {e}")
        return False


def processar_pagina(pagina, arquivo: str, caminho_arquivo: str, pagina_index: int) -> dict:
    texto_pagina = pagina.get_text().strip()
    pagina_ocr = False 

    if len(texto_pagina) < 50:
        pagina_ocr = True
        pix = pagina.get_pixmap(dpi=300)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        img_processada = pre_processar_img(img_array)
        texto_ocr = ocr_img(img_processada)

        if texto_ocr and texto_ocr.strip():
            texto_pagina = texto_ocr

    if texto_pagina.strip():
        texto_limpo = limpar_texto(texto_pagina, is_ocr=pagina_ocr)
        return {
            "text": texto_limpo,
            "metadata": {
                "name_file": arquivo,
                "path_file": caminho_arquivo,
                "page_label": str(pagina_index + 1),
                "page_ocr": pagina_ocr
            }
        }
    
    return None


def processar_pdf(caminho_arquivo: str, arquivo: str) -> list:
    docs = []
    try:
        with pymupdf.open(caminho_arquivo) as documento:
            for pagina_index, pagina in enumerate(documento):
                doc_llama = processar_pagina(pagina, arquivo, caminho_arquivo, pagina_index)
                if doc_llama:
                    docs.append(doc_llama)
    except Exception as e:
        print(f"Erro ao processar o arquivo {arquivo}: {e}")
    return docs


def extrair_texto_pdf(diretorio: str, caminho_json: str = None) -> list:
    print(f"Iniciando extração de texto dos PDFs da pasta: {diretorio}")
    documentos_processados = []
    for arquivo in os.listdir(diretorio):
        if arquivo.lower().endswith(".pdf"):
            caminho_arquivo = os.path.join(diretorio, arquivo)
            documentos_processados.extend(processar_pdf(caminho_arquivo, arquivo))

    print(f"Total de páginas processadas: {len(documentos_processados)}")

    if caminho_json:
        salvar_json(documentos_processados, caminho_json)
    return documentos_processados