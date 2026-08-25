import torch
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from pathlib import Path
import json

torch.set_num_threads(6)

model_id = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

tokenizer = AutoTokenizer.from_pretrained(model_id)

modelo_onnx = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)

classificador = pipeline("zero-shot-classification", model=modelo_onnx, tokenizer=tokenizer)

labels_tipo = [
    "Lei: Atos do Poder Legislativo.",
    "Portaria: Atos administrativos expedidos por órgãos ou autoridades executivas.", 
    "Resolução: Atos normativos expedidos por conselhos ou colegiados."
]

labels_tema = [
    "Educação: Diretrizes curriculares, ensino médio, bases da educação nacional e pesquisa institucional.",
    "Administração Pública: Normas de processos administrativos federais e comissões patrimoniais de conselhos.",  
    "Assistência Social: Gestão e repasses do Programa Bolsa Família e CadÚnico.",  
    "Trânsito e Mobilidade: Regulamentação de veículos, mobilidade e fiscalização por videomonitoramento.",  
    "Saúde e Medicina: Normas sobre perícia médica, telemedicina e regras para coordenação de cursos de graduação em medicina."
]

def classificar_texto(texto: str) -> dict:
    try:
        texto_limpo = " ".join(texto[:1000].split())[:800]

        if not texto_limpo:
            return {}

        template_pt = "Este documento trata de {}."

        resultado_tipo = classificador(
            texto_limpo, 
            labels_tipo, 
            multi_label=False,
            hypothesis_template=template_pt
        )

        resultado_tema = classificador(
            texto_limpo, 
            labels_tema, 
            multi_label=False,
            hypothesis_template=template_pt
        )

        return {
            "tipo_predito": resultado_tipo['labels'][0].split(":")[0],
            "confianca_tipo": round(resultado_tipo['scores'][0], 4),
            "tema_predito": resultado_tema['labels'][0].split(":")[0],
            "confianca_tema": round(resultado_tema['scores'][0], 4)
        }
        
    except Exception as e:
        print(f"Erro no processo de classificação: {e}")
        return {}


def carregar_json(json_caminho: str) -> list:
    try:
        with open(json_caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar o JSON: {e}")
        return []


def processar_classificacao(caminho_entrada: str, caminho_saida: str) -> bool:
    documentos = carregar_json(caminho_entrada)

    if not documentos:
        return False

    documentos_classificados = {}

    print(f"Iniciando classificação. Total de páginas extraídas: {len(documentos)}")

    for documento in documentos:
        nome_arquivo = documento.get("metadata", {}).get("name_file")
        texto = documento.get("text", "")

        if nome_arquivo not in documentos_classificados and texto.strip():
            print(f"Classificando documento: {nome_arquivo}...")
            documentos_classificados[nome_arquivo] = classificar_texto(texto)

        if nome_arquivo in documentos_classificados:
            documento["metadata"].update(documentos_classificados[nome_arquivo])

    try:
        json_saida = Path(caminho_saida)
        json_saida.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_saida, 'w', encoding='utf-8') as f:
            json.dump(documentos, f, ensure_ascii=False, indent=4)
            
        print(f"JSON classificado salvo em: {json_saida}")
        return True
        
    except Exception as e:
        print(f"Erro ao salvar o JSON classificado: {e}")
        return False