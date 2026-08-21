from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

labels_tipo = [
    "Lei Federal (aprovada pelo Congresso)",
    "Portaria de Gestão Interna (órgão, instituto ou conselho)",
    "Resolução Normativa (conselho federal ou de trânsito)"
]

labels_tema = [
    "Trânsito e Mobilidade Urbana",
    "Diretrizes da Educação Nacional",
    "Ética e Regulação da Prática Médica",
    "Direito Administrativo e Processo Público",
    "Assistência Social e Transferência de Renda"
]

vetores_tipo = modelo.encode(labels_tipo, convert_to_tensor=True)
vetores_tema = modelo.encode(labels_tema, convert_to_tensor=True)


def classificar_texto(texto: str) -> tuple:
    """
    Classifica um texto identificando seu Tipo e Tema via similaridade semântica.

    Args:
        texto (str): O conteúdo textual a ser classificado.

    Returns:
        tuple: Uma tupla contendo duas strings no formato (tipo, tema).
    """

    if not texto:
        return "Desconhecido", "Desconhecido"
    try:
        vetor_texto = modelo.encode(texto, convert_to_tensor=True)
        
        similaridade_tipo = modelo.similarity(vetor_texto, vetores_tipo) 
        indice_tipo = similaridade_tipo.argmax().item()
        tipo = labels_tipo[indice_tipo]
        
        similaridade_tema = modelo.similarity(vetor_texto, vetores_tema)
        indice_tema = similaridade_tema.argmax().item()
        tema = labels_tema[indice_tema]
        
        return tipo, tema
    
    except Exception as erro:
        print(f"Erro na classificação: {erro}")
        return "Erro", "Erro"


def processar_classificacao(documentos: list) -> list:    
    docs = {}
    for pagina in documentos:
        nome = pagina.metadata.get("file_name", "Desconhecido")

        if nome not in docs:
            docs[nome] = []

        docs[nome].append(pagina)

    documentos_classificados = []

    for nome_arquivo, paginas in docs.items():
        texto_amostra = ""
        for pagina in paginas:
            texto_amostra += pagina.text + " "
            if len(texto_amostra) >= 2000:
                break 

        tipo, tema = classificar_texto(texto_amostra)
        
        for pagina in paginas:
            pagina.metadata["tipo_documento"] = tipo
            pagina.metadata["tema_documento"] = tema
            documentos_classificados.append(pagina)

    print("Classificação de texto concluída.")
    return documentos_classificados