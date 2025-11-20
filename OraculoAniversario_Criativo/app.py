# Luan Alysson de Souza

# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime
import gradio as gr
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
import shutil
import random

# Carregar .env
load_dotenv()

PDF_PATH = "aniversarios.pdf"
CAPA_IMG = "aniversariocapa.png"
MODELO_EMBED = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Verificar arquivos
assert os.path.exists(PDF_PATH), f"❌ PDF não encontrado: {PDF_PATH}"
assert os.path.exists(CAPA_IMG), f"❌ Imagem de capa não encontrada: {CAPA_IMG}"

historico = []
data_para_doc = {}
TEMP_DIR = "./chroma_temp"
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

def processar_pdf():
    print("📘 Processando PDF...")
    leitor = PdfReader(PDF_PATH)
    documentos = []

    for num, pagina in enumerate(leitor.pages):
        texto = pagina.extract_text()
        if not texto:
            continue

        data_match = re.search(r'(\d{1,2}\s*(?:a\s*\d{1,2}\s*)?de\s+\w+)', texto, re.IGNORECASE)
        data_str = data_match.group().strip() if data_match else f"Página {num+1}"

        doc = Document(
            page_content=texto,
            metadata={
                "fonte": PDF_PATH,
                "data": data_str,
                "pagina": num + 1,
                "conteudo_completo": texto
            }
        )

        documentos.append(doc)

        chave = data_str.lower()
        if "de" in chave:
            data_para_doc[chave] = doc

    print("🔍 Criando índice vetorial com Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBED)
    return Chroma.from_documents(documentos, embeddings, persist_directory=TEMP_DIR)

banco_dados = processar_pdf()

def buscar_data_exata(data_str):
    try:
        if not re.match(r'^\d{1,2}/\d{1,2}$', data_str):
            return None, "⚠️ Use o formato DD/MM (ex: 25/06)"

        dia, mes = map(int, data_str.split('/'))
        if not 1 <= dia <= 31 or not 1 <= mes <= 12:
            return None, "⚠️ Data inválida! Dia (1-31) e Mês (1-12)"

        meses = ["janeiro","fevereiro","março","abril","maio","junho",
                 "julho","agosto","setembro","outubro","novembro","dezembro"]
        data_formatada = f"{dia} de {meses[mes-1]}"

        chave = data_formatada.lower()
        if chave in data_para_doc:
            doc = data_para_doc[chave]
            historico.append((data_formatada, doc))
            return doc, None

        docs = banco_dados.similarity_search(data_formatada, k=3)
        if docs:
            historico.append((data_formatada, docs[0]))
            return docs[0], None

        mensagens = [
            f"😕 Ops! Não encontramos uma entrada específica para **{data_formatada}**.",
            f"📭 Nenhuma menção direta ao dia **{data_formatada}**.",
            f"🔍 Não achamos algo para **{data_formatada}**. Tente outra data.",
            f"🙃 O dia **{data_formatada}** não está disponível."
        ]
        return None, random.choice(mensagens)

    except Exception as e:
        print(f"❌ Erro buscar_data_exata: {str(e)}")
        return None, f"❌ Erro inesperado: {str(e)}"

def formatar_saida(doc):
    conteudo = doc.metadata.get("conteudo_completo", "").strip()

    # Agrupar por seções principais com regex
    secoes = re.split(r'(?=\n(?:OS DIAS DO ANO|GÊMEOS|LEÃO|NÚMEROS E PLANETAS|TARÔ|SAÚDE|CONSELHO|MEDITAÇÃO)\b)', conteudo)
    markdown = f"## 📅 {doc.metadata['data']}  \n**Página:** {doc.metadata['pagina']}\n\n---\n"

    for secao in secoes:
        linhas = [linha.strip() for linha in secao.strip().splitlines() if linha.strip()]
        if not linhas:
            continue

        titulo = linhas[0].upper()
        corpo = " ".join(linhas[1:]).strip()

        if "DIAS DO ANO" in titulo:
            markdown += f"\n### 🗓️ {titulo}\n{corpo}\n"
        elif "NÚMEROS E PLANETAS" in titulo:
            markdown += f"\n### ☀️ {titulo}\n{corpo}\n"
        elif "TARÔ" in titulo:
            markdown += f"\n### 🎴 {titulo}\n{corpo}\n"
        elif "SAÚDE" in titulo:
            markdown += f"\n### 🧬 {titulo}\n{corpo}\n"
        elif "CONSELHO" in titulo:
            markdown += f"\n### 💡 {titulo}\n> {corpo}\n"
        elif "MEDITAÇÃO" in titulo:
            markdown += f"\n### 🧘 {titulo}\n> {corpo}\n"
        else:
            markdown += f"\n### {titulo}\n{corpo}\n"

    markdown += ("\n---\n📘 *Gerado por Luan Alysson com base no livro “A Linguagem Secreta dos Aniversários” de "
                 "Gary Goldschneider e Joost Elffers.*")
    return markdown

def gerar_audio(texto):
    try:
        if not texto.strip():
            return None
        tts = gTTS(text=texto, lang='pt', tld='com.br')
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        return temp_audio.name
    except Exception as e:
        print(f"❌ Erro gerar_audio: {str(e)}")
        return None

def gerar_resposta(data_str):
    doc, erro = buscar_data_exata(data_str)
    if erro:
        return erro, None

    try:
        texto = doc.metadata.get("conteudo_completo", "").strip()
        if not texto:
            return "⚠️ O conteúdo desta data está vazio ou ilegível.", None

        resposta = formatar_saida(doc)
        audio_path = gerar_audio(texto)
        return resposta, audio_path

    except Exception as e:
        print(f"❌ Erro gerar_resposta: {str(e)}")
        return f"⚠️ Houve um erro: {str(e)}", None

def modo_surpresa():
    hoje = datetime.now()
    data_str = f"{hoje.day}/{hoje.month}"
    return gerar_resposta(data_str)

def mostrar_historico():
    if not historico:
        return "ℹ️ Nenhuma busca feita ainda."
    return "\n\n".join([f"✅ {d[0]} - Página {d[1].metadata['pagina']}" for d in historico[::-1]])

with gr.Blocks(title="Aniversário AI") as app:
    gr.Markdown("# 🎉 Explorador de Aniversários em PDF")

    with gr.Tab("🔎 Buscar por Data"):
        entrada = gr.Textbox(label="Digite a data (ex: 25/06)")
        btn = gr.Button("Buscar")

        with gr.Row():
            with gr.Column(scale=3):
                saida = gr.Markdown()
                audio = gr.Audio(label="🔊 Leitura do Conteúdo", autoplay=False)
            with gr.Column(scale=1):
                imagem = gr.Image(value=CAPA_IMG, label="Capa do Livro")

        btn.click(gerar_resposta, inputs=entrada, outputs=[saida, audio])

    with gr.Tab("🎁 Aniversariante do Dia"):
        btn_surpresa = gr.Button("Me surpreenda hoje!")
        saida_s = gr.Markdown()
        audio_s = gr.Audio(label="🔊 Leitura", autoplay=False)
        btn_surpresa.click(modo_surpresa, outputs=[saida_s, audio_s])

    with gr.Tab("🕘 Histórico de Buscas"):
        btn_hist = gr.Button("Ver Histórico")
        historico_out = gr.Textbox(label="Buscas Recentes", lines=10)
        btn_hist.click(mostrar_historico, outputs=historico_out)

if __name__ == "__main__":
    app.launch()
