import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import gradio as gr

# Carrega a chave do .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
# Verifica se a chave está disponível
if not api_key:
    raise ValueError("❌ Variável OPENROUTER_API_KEY não encontrada.")

# Define as variáveis que o LangChain espera
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Instancia o modelo
llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    temperature=0.5
)

# Função simples sem histórico
def responder(mensagem):
    try:
        resposta = llm.invoke(mensagem)
        return resposta.content
    except Exception as e:
        import traceback
        return f"❌ Erro:\n{traceback.format_exc()}"

# Interface Gradio simples
app = gr.Interface(
    fn=responder,
    inputs=gr.Textbox(placeholder="Digite sua pergunta aqui", label="Mensagem"),
    outputs=gr.Textbox(label="Resposta do Chatbot"),
    title="Meu Primeiro Chatbot com IA Generativa",
    description="Teste do modelo DeepSeek via OpenRouter com retorno direto.",
)

app.launch(share=True)