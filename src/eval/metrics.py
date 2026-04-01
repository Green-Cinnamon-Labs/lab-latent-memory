"""
Métricas de avaliação.

Compara yₜ_full (baseline com histórico completo) contra
yₜ_mem (experimental com memória persistente).

O que queremos medir:
1. O modelo com mₜ lembra fatos ditos antes?
2. Ele resolve referências ("isso", "o que eu disse")?
3. Ele mantém restrições estabelecidas na conversa?
4. A resposta é coerente com o tom e direção da conversa?
"""

from dataclasses import dataclass


@dataclass
class EvalCase:
    """Um caso de teste."""
    conversation: list       # lista de mensagens até o momento
    query: str               # nova pergunta que depende do histórico
    expected_elements: list   # fatos/conceitos que a resposta deve conter
    category: str             # fact_retention | reference | constraint | coherence


@dataclass
class EvalResult:
    """Resultado de um caso."""
    case: EvalCase
    response_full: str       # resposta do baseline
    response_mem: str        # resposta do experimental
    score_full: float        # 0-1
    score_mem: float         # 0-1


def score_fact_retention(response: str, expected_elements: list) -> float:
    """
    Verifica se os elementos esperados aparecem na resposta.
    
    Versão v0: busca textual simples.
    Versão futura: embedding similarity ou julgamento por LLM.
    """
    if not expected_elements:
        return 1.0
    found = sum(1 for e in expected_elements if e.lower() in response.lower())
    return found / len(expected_elements)


# ---- Benchmark de exemplo ----

SAMPLE_BENCHMARK = [
    EvalCase(
        conversation=[
            {"role": "user", "content": "Meu nome é Matheus e eu trabalho com engenharia."},
            {"role": "assistant", "content": "Prazer, Matheus! Como posso te ajudar?"},
        ],
        query="Qual é meu nome e o que eu faço?",
        expected_elements=["Matheus", "engenharia"],
        category="fact_retention",
    ),
    EvalCase(
        conversation=[
            {"role": "user", "content": "Estou pesquisando compressão de contexto para LLMs."},
            {"role": "assistant", "content": "Interessante. Que abordagem você está considerando?"},
            {"role": "user", "content": "Um estado latente persistente que substitui o replay de tokens."},
            {"role": "assistant", "content": "Entendi. Isso é diferente de RAG."},
        ],
        query="Resuma o que estou pesquisando.",
        expected_elements=["estado latente", "persistente", "contexto"],
        category="fact_retention",
    ),
    EvalCase(
        conversation=[
            {"role": "user", "content": "Só me responda em português, nunca em inglês."},
            {"role": "assistant", "content": "Combinado, só português."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "A capital da França é Paris."},
        ],
        query="What is 2 + 2?",
        expected_elements=["4"],  # mas deve responder em PT
        category="constraint",
    ),
]
