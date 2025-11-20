# Autor: Luan Alysson de Souza

# -*- coding: utf-8 -*-
"""app"""

import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Carrega variáveis de ambiente
load_dotenv()
OPENROUTER_API_KEY = os.getenv("ROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("❌ A variável de ambiente ROUTER_API_KEY não está definida. Verifique o arquivo .env.")

# Embedding robusto
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
qa_chain = None
processed_file = None

# Carrega o PDF fixo automaticamente
def load_default_pdf():
    global qa_chain, processed_file
    try:
        loader = PyPDFLoader("LegisMiner.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, embeddings)

        llm = ChatOpenAI(
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            model="mistralai/mistral-7b-instruct:free",
            temperature=0.3
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        processed_file = "LegisMiner.pdf"
        print("✅ LegisMiner.pdf carregado automaticamente.")
    except Exception as e:
        print(f"❌ Erro ao carregar LegisMiner.pdf automaticamente: {e}")

def calculate_rag_metrics(query, response, source_docs):
    metrics = {}
    try:
        query_embedding = embeddings.embed_query(query)
        response_embedding = embeddings.embed_query(response)
        metrics["query_response_similarity"] = cosine_similarity(
            [query_embedding], [response_embedding]
        )[0][0]

        # Mantido apenas para fins internos, mas não será exibido
        doc_similarities = []
        for doc in source_docs:
            doc_embedding = embeddings.embed_query(doc.page_content[:1000])
            similarity = cosine_similarity([response_embedding], [doc_embedding])[0][0]
            doc_similarities.append(similarity)

        metrics["avg_response_source_similarity"] = np.mean(doc_similarities) if doc_similarities else 0
        metrics["max_response_source_similarity"] = max(doc_similarities) if doc_similarities else 0
        metrics["num_source_documents"] = len(source_docs)

    except Exception as e:
        metrics["error"] = str(e)

    return metrics

def process_pdf(file):
    global qa_chain, processed_file

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file)
        pdf_path = tmp.name

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, embeddings)

        llm = ChatOpenAI(
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            model="mistralai/mistral-7b-instruct:free",
            temperature=0.3
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        processed_file = os.path.basename(pdf_path)
        return f"✅ PDF processado com sucesso: {processed_file}"

    except Exception as e:
        return f"❌ Erro ao processar PDF: {str(e)}"

def ask_question(question):
    global qa_chain

    if qa_chain is None:
        return "⚠️ Por favor, carregue um PDF primeiro", "", ""

    try:
        system_prompt = (
            "Você é um assistente especialista em mineração, legislação ambiental e políticas públicas. "
            "Seu papel é responder perguntas com base no conteúdo do PDF carregado, que trata do ambiente regulatório da mineração. "
            "Siga estas instruções com rigor:\n\n"
            "1. A resposta deve estar no mesmo idioma em que a pergunta foi feita.\n"
            "2. Utilize apenas as informações contidas no PDF como base.\n"
            "3. Nunca omita dados relevantes encontrados no conteúdo original.\n"
            "4. Mencione, sempre que possível, trechos, leis, datas ou tópicos do PDF usados como base.\n"
            "5. Se a pergunta for técnica, use linguagem técnica. Se for simples, explique de forma acessível.\n"
            "6. Caso a resposta exija algo que não está no PDF, diga claramente: "
            "\"Com base no conteúdo fornecido, não há informação direta sobre este ponto específico.\"\n"
            "7. Seja objetivo, preciso e fiel ao conteúdo carregado.\n\n"
            "Agora, responda a pergunta abaixo com base no PDF:"
        )

        resposta = qa_chain.invoke({
            "query": f"{system_prompt}\n\nPergunta: {question}"
        })

        sources = "\n\n".join(
            [f"📄 Fonte {i+1}:\n{doc.page_content[:500]}..."
             for i, doc in enumerate(resposta['source_documents'])]
        )

        metrics = calculate_rag_metrics(
            question,
            resposta['result'],
            resposta['source_documents']
        )

        # Exibe apenas o nível de confiança
        confidence = metrics.get("query_response_similarity", 0)
        metrics_text = f"🔎 Nível de Confiança da Resposta: {confidence:.2f}"

        return resposta['result'], sources, metrics_text

    except Exception as e:
        return f"❌ Erro ao processar pergunta: {str(e)}", "", ""

# Interface Gradio
with gr.Blocks(title="Chat com PDF usando RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Artificial Intelligence Applied to Regulatory Standard Processing in Mining\n###  Development of a Decision Support Tool")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📤 Envie um PDF", type="binary")
            process_btn = gr.Button("Processar PDF", variant="primary")
            status_output = gr.Textbox(label="Status")

        with gr.Column(scale=2):
            question_input = gr.Textbox(label="Faça uma pergunta sobre Normas da Mineração", lines=3)
            ask_btn = gr.Button("Enviar Pergunta", variant="primary")
            answer_output = gr.Textbox(label="✅ Resposta", interactive=False)

        with gr.Accordion("📄 Fontes usadas", open=False):
            sources_output = gr.Textbox(label="Trechos relevantes", lines=10)

        with gr.Accordion("📊 Métricas RAG", open=False):
            metrics_output = gr.Textbox(label="Métricas", lines=4)

    process_btn.click(
        fn=process_pdf,
        inputs=file_input,
        outputs=status_output
    )

    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output, metrics_output]
    )

# Carrega o PDF fixo ao iniciar
load_default_pdf()

# Compartilhamento opcional
share = True if 'COLAB_JUPYTER_TRANSPORT' in os.environ else False
demo.launch(share=share, debug=False)

