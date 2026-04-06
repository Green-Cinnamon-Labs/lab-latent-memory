"""
Implementações concretas do operador G.

Cada uma representa uma hipótese diferente sobre como
a memória deve ser atualizada.

EMA: a mais simples. Mistura linear do estado anterior com a nova entrada.
GRU: porta de esquecimento aprendida. Decide quanto do passado manter.
MLP: transformação não-linear livre. Mais expressiva, mais difícil de controlar.

Conforme você estudar mais matemática (operadores, dinâmica, estabilidade),
novos updaters aparecerão aqui.
"""

import torch
import torch.nn as nn
from .base import Updater


class EMAUpdater(Updater):
    """
    Exponential Moving Average.
    
    mₜ₊₁ = (1 - α) · mₜ + α · eₜ
    
    α controla velocidade de esquecimento.
    - α pequeno → memória longa, muda devagar
    - α grande  → memória curta, reage rápido
    
    Sem parâmetros treináveis. Bom para v0.
    Matematicamente: é um filtro linear de primeira ordem.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def __call__(self, m_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * m_t + self.alpha * e_t

    def state_dict(self):
        return {"alpha": self.alpha}

    def load_state_dict(self, d):
        self.alpha = d["alpha"]


class GRUUpdater(Updater):
    """
    Gated Recurrent Unit como operador de atualização.
    
    A GRU tem portas (gates) que aprendem quando esquecer e quando incorporar.
    É a versão aprendida de "quanto do passado manter".
    
    mₜ₊₁ = GRU(mₜ, eₜ)
    
    Internamente:
      zₜ = σ(Wz · [mₜ, eₜ])          # porta de atualização
      rₜ = σ(Wr · [mₜ, eₜ])          # porta de reset
      m̃  = tanh(W · [rₜ ⊙ mₜ, eₜ])  # candidato
      mₜ₊₁ = (1 - zₜ) ⊙ mₜ + zₜ ⊙ m̃  # mistura
    
    Tem parâmetros treináveis → precisa de dados para ajustar.
    """

    def __init__(self, memory_dim: int, input_dim: int):
        self.gru = nn.GRUCell(input_size=input_dim, hidden_size=memory_dim)

    def __call__(self, m_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        # GRUCell espera (batch, features)
        m_batch = m_t.unsqueeze(0) if m_t.dim() == 1 else m_t
        e_batch = e_t.unsqueeze(0) if e_t.dim() == 1 else e_t
        m_next = self.gru(e_batch, m_batch)
        return m_next.squeeze(0)

    def state_dict(self):
        return self.gru.state_dict()

    def load_state_dict(self, d):
        self.gru.load_state_dict(d)


class VectorFieldUpdater(Updater):
    """
    Campo vetorial como operador de atualização (pivot vector-field memory).

    mₜ₊₁ = mₜ + Δt · F(mₜ, eₜ; θ)

    F é uma pequena rede que define um campo vetorial sobre o espaço de memória.
    A memória não "armazena" experiências — o campo F é deformado por elas,
    criando drains/sources/rotações no espaço que enviesam trajetórias futuras.

    Δt controla o tamanho do passo de integração (estabilidade vs velocidade).
    norm_clip previne divergência do vetor de memória.

    Hipótese central a testar:
      Queries similares submetidas ao mesmo campo F devem convergir para
      regiões similares do espaço (trajectory consistency).
    """

    def __init__(self, memory_dim: int, input_dim: int, dt: float = 0.1, norm_clip: float = 10.0):
        self.dt = dt
        self.norm_clip = norm_clip
        # F: campo vetorial parametrizado — recebe estado atual + entrada, retorna direção
        self.field = nn.Sequential(
            nn.Linear(memory_dim + input_dim, memory_dim),
            nn.Tanh(),
            nn.Linear(memory_dim, memory_dim),
        )

    def __call__(self, m_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([m_t, e_t], dim=-1)
        direction = self.field(combined)           # F(mₜ, eₜ; θ)
        m_next = m_t + self.dt * direction         # passo de integração Euler
        # previne drift: se a norma explodir, reescala
        norm = m_next.norm()
        if norm > self.norm_clip:
            m_next = m_next * (self.norm_clip / norm)
        return m_next

    def state_dict(self):
        return self.field.state_dict()

    def load_state_dict(self, d):
        self.field.load_state_dict(d)


class MLPUpdater(Updater):
    """
    MLP residual como operador.
    
    mₜ₊₁ = mₜ + f([mₜ; eₜ])
    
    Onde f é um MLP pequeno. A conexão residual (mₜ +) garante
    que a memória não mude demais a cada passo.
    
    Mais expressivo que EMA/GRU, mas mais difícil de estabilizar.
    """

    def __init__(self, memory_dim: int, input_dim: int, hidden_dim: int = 256):
        self.net = nn.Sequential(
            nn.Linear(memory_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def __call__(self, m_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([m_t, e_t], dim=-1)
        delta = self.net(combined)
        return m_t + delta  # residual

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, d):
        self.net.load_state_dict(d)
