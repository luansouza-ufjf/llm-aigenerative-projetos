import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import gradio as gr

# Carrega a chave da API do arquivo .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Verifica se a chave foi carregada
if not api_key:
    raise ValueError("❌ Variável OPENROUTER_API_KEY não encontrada.")

# Configurações do LangChain para o OpenRouter
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Instancia o modelo LLM
llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    temperature=0.7
)

# Função principal que gera recomendações de negócios
def responder(mensagem):
    try:
        prompt_inicial = (
            "Você é um consultor de negócios especializado em ajudar empreendedores a desenvolver ideias, melhorar estratégias e tomar decisões com base em boas práticas de gestão. "
            "Com base na descrição do cliente sobre seu negócio ou ideia, ofereça conselhos práticos e objetivos, sugerindo possíveis estratégias, melhorias ou ferramentas úteis.\n\n"
            "Empreendedor: " + mensagem + "\n"
            "Recomendações:"
        )
        resposta = llm.invoke(prompt_inicial)
        return resposta.content
    except Exception as e:
        import traceback
        return f"❌ Erro:\n{traceback.format_exc()}"

# Interface Gradio
app = gr.Interface(
    fn=responder,
    inputs=gr.Textbox(
        placeholder="Ex: Tenho uma loja online de roupas femininas e quero aumentar minhas vendas.",
        label="Descrição do Negócio / Business Description",
        lines=3,
        info="Tecnologia usada: IA com LangChain + OpenRouter para consultoria de negócios / AI with LangChain + OpenRouter for Business Consulting"
    ),
    outputs=gr.Textbox(label="Sugestões do Consultor / Consultant Suggestions"),
    title="Consultor de Negócios com IA / AI Business Consultant",
    description="Obtenha ideias, estratégias e conselhos personalizados para o seu negócio com ajuda da inteligência artificial. / Get tailored business ideas, strategies, and advice using AI.",
)

# Executa a aplicação com link público (útil para testes locais)
app.launch(share=True)
