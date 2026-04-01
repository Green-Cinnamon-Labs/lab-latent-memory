"""
Objeto de memória mₜ e interface base para operadores de atualização G.

mₜ é um vetor em ℝⁿ que acumula informação relevante da conversa.
G é a função que atualiza mₜ dado uma nova entrada.

A pergunta central: qual G preserva o que importa e esquece o que não importa?
"""

import torch
import json
from abc import ABC, abstractmethod
from pathlib import Path


class MemoryState:
    """
    O objeto mₜ. É um vetor persistente em ℝⁿ.
    
    Pode ser salvo em disco e recarregado entre interações.
    Cada interação produz um novo estado — a sequência m₀, m₁, m₂, ...
    é a "trajetória de memória" da conversa.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vector = torch.zeros(dim)  # m₀ = vetor zero
        self.step = 0                   # contador de atualizações

    def update(self, new_vector: torch.Tensor):
        """Substitui o vetor interno. Chamado após G computar mₜ₊₁."""
        assert new_vector.shape == (self.dim,), \
            f"Esperado ({self.dim},), recebeu {new_vector.shape}"
        self.vector = new_vector.detach()
        self.step += 1

    def save(self, path: str):
        """Salva mₜ em disco."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "vector": self.vector,
            "step": self.step,
            "dim": self.dim,
        }, p)

    @classmethod
    def load(cls, path: str) -> "MemoryState":
        """Carrega mₜ do disco."""
        data = torch.load(path, weights_only=True)
        state = cls(dim=data["dim"])
        state.vector = data["vector"]
        state.step = data["step"]
        return state

    def __repr__(self):
        norm = self.vector.norm().item()
        return f"MemoryState(dim={self.dim}, step={self.step}, |m|={norm:.4f})"


class Updater(ABC):
    """
    Interface base para o operador G.
    
    G: (mₜ, eₜ) → mₜ₊₁
    
    Onde eₜ é o embedding da nova interação (não o texto cru,
    mas uma representação vetorial extraída do modelo).
    """

    @abstractmethod
    def __call__(self, m_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """Computa mₜ₊₁ = G(mₜ, eₜ)"""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Retorna parâmetros treináveis do updater, se houver."""
        ...

    @abstractmethod
    def load_state_dict(self, d: dict):
        """Carrega parâmetros."""
        ...
