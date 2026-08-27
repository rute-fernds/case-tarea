import streamlit as st
from src.config import CAMINHO_DB, ICONE_MODELO, ICONE_USUARIO, FAVICON
from src.chat import iniciar_chat

st.set_page_config(
    page_title="Gemma IA", 
    page_icon= FAVICON,
    layout="centered"
)

st.markdown("## ⭐Gemma: Assistente de Documentos")
st.markdown("##### Consulte e explore seus documentos com o Gemma 2 2B")


@st.cache_resource(show_spinner=False)
def carregar_gemma():
    with st.spinner("Carregando modelo Gemma 2:2b e conectando ao LanceDB..."):
        return iniciar_chat(CAMINHO_DB)

chat_engine = carregar_gemma()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []


def exibir_fontes(fontes):
    if fontes:
        with st.expander("📄 Fontes"):
            for fonte in fontes:
                st.markdown(f"**Arquivo:** {fonte['arquivo']} (Pág: {fonte['pagina']})")
                st.info(f'"{fonte["texto"]}"')


for msg in st.session_state.mensagens:
    avatar = ICONE_USUARIO if msg["role"] == "user" else ICONE_MODELO
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            st.markdown("**Gemma**")
        st.markdown(msg["content"])
        
        if "fontes" in msg:
            exibir_fontes(msg["fontes"])


if pergunta := st.chat_input("Pergunte ao Gemma"):
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar=ICONE_USUARIO):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar=ICONE_MODELO):
        st.markdown("**Gemma**")
        
        with st.spinner("Gerando resposta..."):
            try:
                resposta = chat_engine.chat(pergunta)
                sucesso = True
            except Exception as e:
                sucesso = False
                st.error(f"Erro de comunicação com o LLM: {e}")
        
        if sucesso:
            st.markdown(resposta.response)
            
            fontes_extraidas = []
            if hasattr(resposta, 'source_nodes'):
                for node in resposta.source_nodes:
                    fontes_extraidas.append({
                        "arquivo": node.node.metadata.get('name_file', 'Desconhecido'),
                        "pagina": node.node.metadata.get('page_label', '?'),
                        "texto": node.node.text
                    })
            
            exibir_fontes(fontes_extraidas)

            st.session_state.mensagens.append({
                "role": "assistant", 
                "content": resposta.response,
                "fontes": fontes_extraidas
            })