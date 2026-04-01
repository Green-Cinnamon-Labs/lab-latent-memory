"""
Runner: orquestra uma conversa experimental.

Fluxo de cada turno:

  1. Recebe nova mensagem xₜ
  2. Extrai embedding eₜ do modelo
  3. Atualiza memória: mₜ₊₁ = G(mₜ, eₜ)
  4. Injeta mₜ no modelo via hook
  5. Gera resposta yₜ_mem
  6. (Opcionalmente) gera yₜ_full via baseline
  7. Salva mₜ₊₁ em disco
"""

import torch
from ..model.loader import InstrumentedModel
from ..memory.base import MemoryState, Updater
from ..injection.injector import ActivationInjector


class ExperimentalRunner:
    """
    Executa uma conversa usando memória persistente.
    """

    def __init__(
        self,
        model: InstrumentedModel,
        memory: MemoryState,
        updater: Updater,
        injector: ActivationInjector,
        target_layer: int,
    ):
        self.model = model
        self.memory = memory
        self.updater = updater
        self.injector = injector
        self.target_layer = target_layer

        # Registra o hook de injeção
        self.model.register_write_hook(
            layer_idx=target_layer,
            modifier_fn=self.injector.modifier_fn,
        )

    def extract_embedding(self, text: str) -> torch.Tensor:
        """
        Extrai um embedding da entrada usando o próprio modelo.
        
        Usa o hidden state da última camada, último token,
        como representação da entrada.
        
        Existem formas melhores (mean pooling, camada específica),
        mas essa é funcional para v0.
        """
        self.model.register_read_hook(self.model.num_layers - 1)
        
        inputs = self.model.tokenizer(text, return_tensors="pt").to(
            self.model.model.device
        )
        with torch.no_grad():
            self.model.model(**inputs)

        # Pega ativação da última camada, último token
        last_layer = self.model.num_layers - 1
        hidden = self.model._captured.get(last_layer)
        if hidden is None:
            raise RuntimeError("Hook não capturou ativação")
        
        embedding = hidden[0, -1, :]  # (hidden_dim,)
        return embedding

    def step(self, user_message: str, max_new_tokens: int = 128) -> str:
        """
        Executa um turno da conversa.
        
        1. Extrai eₜ da mensagem
        2. Atualiza mₜ₊₁ = G(mₜ, eₜ)  (nota: pode precisar projetar eₜ para dim de mₜ)
        3. Injeta mₜ no modelo
        4. Gera resposta
        5. Salva mₜ₊₁
        """
        # 1. Extrair embedding da entrada
        e_t = self.extract_embedding(user_message)

        # 2. Atualizar memória
        # NOTA: eₜ vive em ℝᵈ (hidden_dim), mₜ vive em ℝⁿ (memory_dim)
        # Se dims diferem, precisa projetar. Por ora, assume dims iguais.
        m_next = self.updater(self.memory.vector, e_t)
        self.memory.update(m_next)

        # 3. Injetar memória atual no modelo
        self.injector.set_memory(self.memory.vector)

        # 4. Gerar resposta
        response = self.model.generate(user_message, max_new_tokens=max_new_tokens)

        return response
