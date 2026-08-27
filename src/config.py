import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_PDFS = os.path.join(BASE_DIR, 'data', 'raw')
CAMINHO_JSON_EXTRAIDO = os.path.join(BASE_DIR, 'data', 'processed', 'texto_extraido.json')
CAMINHO_JSON_CLASSIFICADO = os.path.join(BASE_DIR, 'data', 'processed', 'texto_classificado.json')
CAMINHO_DB = os.path.join(BASE_DIR, 'lancedb')

ICONE_USUARIO = os.path.join(BASE_DIR, "src", "img", "user.png")
ICONE_MODELO = os.path.join(BASE_DIR, "src", "img", "ia.png")
FAVICON = os.path.join(BASE_DIR, "src", "img", "icone.png")

CAMINHO_TESSERACT = r'C:\Program Files\Tesseract-OCR'

CONFIG_TESSERACT = r'-l por --psm 6'

NUM_THREADS = 6

