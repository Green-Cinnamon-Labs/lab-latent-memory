"""
Experimento v0: Injeção de memória via ativações intermediárias.

Este é o primeiro experimento. O objetivo NÃO é obter resultados bons.
O objetivo é verificar que o pipeline funciona:
  - modelo carrega
  - hooks funcionam
  - mₜ é atualizado
  - injeção acontece sem quebrar a geração
  - respostas são produzidas

Só depois de confirmar que o pipeline roda é que faz sentido
investigar qual G, qual camada, qual projeção funciona melhor.

Para rodar:
  1. Instalar dependências: pip install -r requirements.txt
  2. Baixar um modelo pequeno (phi-2 ou similar)
  3. python -m experiments.v0_activation_inject.run
"""

import sys
sys.path.insert(0, ".")

from src.model.loader import InstrumentedModel
from src.memory.base import MemoryState
from src.memory.updaters import EMAUpdater
from src.injection.injector import ActivationInjector


def main():
    # --- Configuração ---
    MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"  # MHA real, HF+Ollama oficial, ~9.2GB FP32
    MEMORY_DIM = 512
    TARGET_LAYER = 12  # camada do meio (SmolLM2-1.7B tem 24 camadas)

    print("=" * 60)
    print("EXPERIMENTO v0: Injeção de memória em ativações")
    print("=" * 60)

    # 1. Carregar modelo
    print(f"\n[1] Carregando modelo {MODEL_NAME}...")
    model = InstrumentedModel(MODEL_NAME, device="auto")
    model.load()
    print(f"    Modelo carregado: {model.num_layers} camadas, hidden_dim={model.hidden_dim}")

    # 2. Criar memória
    # NOTA: para v0, memory_dim == hidden_dim para evitar projeção extra
    memory = MemoryState(dim=model.hidden_dim)
    print(f"\n[2] Memória inicializada: {memory}")

    # 3. Criar updater
    updater = EMAUpdater(alpha=0.1)
    print(f"\n[3] Updater: EMA(α=0.1)")

    # 4. Criar injector
    injector = ActivationInjector(
        memory_dim=model.hidden_dim,
        hidden_dim=model.hidden_dim,
        scale=0.1,
    )
    print(f"\n[4] Injector: ativação na camada {TARGET_LAYER}")

    # 5. Registrar hooks
    model.register_write_hook(TARGET_LAYER, injector.modifier_fn)
    model.register_read_hook(model.num_layers - 1)  # para extrair embeddings
    print(f"\n[5] Hooks registrados")

    # 6. Conversa de teste
    conversation = [
        "Meu nome é Matheus e eu sou engenheiro.",
        "Eu estou pesquisando compressão de contexto para LLMs.",
        "Qual é meu nome e o que eu pesquiso?",
    ]

    print(f"\n[6] Rodando conversa de teste ({len(conversation)} turnos)...\n")
    print("-" * 60)

    for i, msg in enumerate(conversation):
        print(f"\n>> USER (turno {i}): {msg}")

        # Extrair embedding
        inputs = model.tokenizer(msg, return_tensors="pt").to(model.model.device)
        import torch
        with torch.no_grad():
            model.model(**inputs)

        last_layer = model.num_layers - 1
        hidden = model._captured.get(last_layer)
        if hidden is not None:
            e_t = hidden[0, -1, :]  # embedding do último token

            # Atualizar memória
            m_next = updater(memory.vector.to(e_t.device), e_t)
            memory.update(m_next.cpu())

            # Setar memória no injector
            injector.set_memory(memory.vector.to(model.model.device))

        # Gerar resposta (sem histórico — só a mensagem atual + memória injetada)
        response = model.generate(msg, max_new_tokens=100)
        print(f"<< MODEL: {response}")
        print(f"   Memória: {memory}")

    print("\n" + "-" * 60)
    print("\nExperimento v0 concluído.")
    print("Se chegou aqui sem erro, o pipeline funciona.")
    print("Próximo passo: comparar com baseline full-context.")

    # Salvar memória final
    memory.save("data/memory_states/v0_final.pt")
    print(f"\nMemória salva em data/memory_states/v0_final.pt")


if __name__ == "__main__":
    main()
