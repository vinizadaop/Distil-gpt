🤖 Gerador de Respostas com Modelos de Linguagem (Falcon 7B & Zephyr 7B)
Este projeto permite carregar modelos de linguagem poderosos da Hugging Face (como o Falcon 7B Instruct e o Zephyr 7B Beta) e gerar respostas a partir de prompts fornecidos pelo usuário. A interação pode ser feita via terminal ou através de uma interface com Gradio (dependendo da versão do script usada).

🚀 Modelos Suportados
tiiuae/falcon-7b-instruct

HuggingFaceH4/zephyr-7b-beta

📦 Requisitos
Python 3.8+

torch

transformers

gradio (opcional)

Instale as dependências com:

bash
Copiar
Editar
pip install torch transformers gradio
⚙️ Como Usar
1. Modo Interativo via Terminal (Zephyr 7B)
Este script carrega o modelo Zephyr-7B-Beta e permite enviar prompts diretamente pelo terminal:

bash
Copiar
Editar
python zephyr_terminal.py
Exemplo de uso:
bash
Copiar
Editar
Digite seu prompt (ou 'sair' para encerrar): O que é aprendizado de máquina?
Resposta gerada:
Aprendizado de máquina é um campo da inteligência artificial que...
2. Modo Interface com Gradio (Falcon 7B)
Uma interface web interativa pode ser implementada com gr.Interface (veja gradio_falcon.py, por exemplo). O código já inclui carregamento do Falcon 7B Instruct e pode ser adaptado assim:

python
Copiar
Editar
import gradio as gr

def interface_gradio():
    model, tokenizer = carregar_modelo("tiiuae/falcon-7b-instruct")

    def responder(pergunta):
        return gerar_resposta(pergunta, model, tokenizer)

    gr.Interface(fn=responder, inputs="text", outputs="text").launch()

interface_gradio()
🧠 Funções Principais
carregar_modelo(model_name): Carrega o modelo e tokenizer, com suporte automático para GPU se disponível.

gerar_resposta(prompt, model, tokenizer): Gera uma resposta para o prompt fornecido.

main(): Loop de entrada para interação com o usuário via terminal.

💻 Dispositivos Suportados
O código detecta automaticamente se há uma GPU disponível e utiliza float16 quando aplicável para acelerar a inferência.

🛠️ Problemas Conhecidos
Alguns modelos podem exigir mais de 16 GB de VRAM para carregar com eficiência.

Certifique-se de que seu ambiente Python é compatível com a versão de torch e transformers necessária.
