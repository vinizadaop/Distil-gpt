import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Modelo
MODEL_NAME = "tiiuae/falcon-7b-instruct"

# Verificação de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Carrega modelo e tokenizer
print("Carregando o modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
model.to(device)
print("Modelo carregado com sucesso.")

# Define pad_token se necessário
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Função de geração otimizada
def gerar_resposta(pergunta):
    prompt = (
        f"Você é um assistente inteligente e educado. "
        f"Responda com clareza e sem repetir palavras. Responda em português do Brasil.\n"
        f"Pergunta: {pergunta.strip()}\nResposta:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.5,              # Menor temperatura = mais foco
            top_p=0.85,                  # Limita aleatoriedade
            repetition_penalty=1.4,      # Penaliza repetições
            no_repeat_ngram_size=3,      # Evita repetir sequências de 3 palavras
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    resposta = tokenizer.decode(output[0], skip_special_tokens=True)

    # Limpa a resposta (remove o prompt, repetições ou lixo)
    if "Resposta:" in resposta:
        resposta = resposta.split("Resposta:")[-1]

    # Remove duplicações simples
    linhas = resposta.split("\n")
    linhas_unicas = []
    for linha in linhas:
        if linha.strip() not in linhas_unicas:
            linhas_unicas.append(linha.strip())
    resposta_final = "\n".join(linhas_unicas).strip()

    return resposta_final if resposta_final else "Desculpe, não consegui gerar uma resposta clara."

# Interface Gradio
gr.Interface(
    fn=gerar_resposta,
    inputs=gr.Textbox(lines=5, label="Digite sua pergunta"),
    outputs=gr.Textbox(label="Resposta do Assistente"),
    title="distil gpt – Assistente em Português",
    description="Assistente IA otimizado para gerar respostas sem repetição e em português do Brasil."
).launch()
