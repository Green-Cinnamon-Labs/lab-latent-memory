"""
Injeção de mₜ na computação do transformer.

Este é o módulo mais crítico do projeto. Ele responde:
como o estado de memória entra na computação da LLM?

Porta 2: soft tokens prepended (mₜ projetado como tokens virtuais)
Porta 3: modificação direta de ativações intermediárias

O desafio matemático: mₜ vive em ℝⁿ (seu espaço de memória),
mas as ativações do transformer vivem em ℝᵈ (hidden_dim do modelo).
Precisa de uma projeção entre os dois espaços.
"""

import torch
import torch.nn as nn


class ActivationInjector:
    """
    Porta 3: injeta mₜ diretamente nas ativações de uma camada.
    
    Estratégia: após a camada target_layer computar seu output,
    somar uma projeção de mₜ ao hidden state do último token.
    
    Por que o último token? Porque no modo causal (autoregressive),
    o último token é onde a "decisão" do próximo token se concentra.
    
    A projeção linear W: ℝⁿ → ℝᵈ mapeia do espaço de memória
    para o espaço do modelo. W é treinável.
    """

    def __init__(self, memory_dim: int, hidden_dim: int, scale: float = 0.1):
        self.projection = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.scale = scale  # controla intensidade da injeção
        self._memory_vector = None

    def set_memory(self, m_t: torch.Tensor):
        """Define o mₜ atual. Chamado antes de cada geração."""
        self._memory_vector = m_t

    def modifier_fn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Função que será registrada como hook no modelo.
        
        hidden_states: (batch, seq_len, hidden_dim)
        
        Modifica o último token somando a projeção de mₜ.
        """
        if self._memory_vector is None:
            return hidden_states

        # Projetar mₜ para o espaço do modelo
        projected = self.projection(self._memory_vector)  # (hidden_dim,)
        
        # Somar ao último token, escalado
        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] += self.scale * projected

        return hidden_states


class KVCacheInjector:
    """
    Porta 3 (variante): injeta mₜ como entradas virtuais na KV-cache.
    
    A KV-cache armazena as chaves (K) e valores (V) de tokens anteriores.
    Ao inserir K e V virtuais derivados de mₜ, é como se o modelo
    "se lembrasse" de tokens que nunca existiram — tokens fantasma
    que codificam a memória.
    
    Isso é mais sutil que a injeção por ativação porque afeta
    o mecanismo de atenção diretamente: o modelo passa a "atender"
    à memória como se fosse parte da sequência.
    
    NOTA: esta implementação é um esqueleto. A interface real da
    KV-cache varia muito entre modelos. Implementação completa
    requer estudo da arquitetura específica do modelo alvo.
    """

    def __init__(self, memory_dim: int, hidden_dim: int, num_heads: int, num_virtual_tokens: int = 4):
        self.num_virtual_tokens = num_virtual_tokens
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads

        # Projeções: mₜ → K virtual e V virtual
        self.key_projection = nn.Linear(memory_dim, num_virtual_tokens * hidden_dim, bias=False)
        self.value_projection = nn.Linear(memory_dim, num_virtual_tokens * hidden_dim, bias=False)

    def compute_virtual_kv(self, m_t: torch.Tensor):
        """
        Dado mₜ, computa K e V virtuais.
        
        Retorna:
            virtual_keys:   (1, num_heads, num_virtual_tokens, head_dim)
            virtual_values: (1, num_heads, num_virtual_tokens, head_dim)
        """
        # TODO: implementar quando a arquitetura-alvo for escolhida
        # A forma exata depende de como o modelo específico organiza sua cache
        raise NotImplementedError(
            "KV-cache injection depende da arquitetura específica do modelo. "
            "Comece pela ActivationInjector e evolua para cá depois de estudar "
            "a implementação interna do modelo alvo."
        )
