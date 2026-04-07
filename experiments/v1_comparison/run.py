"""
Experimento v1: benchmark comparativo de memória.

Três condições para cada caso de teste:
  sem_memoria  — query enviada ao modelo sem histórico e sem injeção
  ema_memoria  — mₜ acumulado via EMA + injeção em ativações (camada TARGET_LAYER)
  full_context — histórico completo reinjetado via Ollama

Métricas:
  score       — fração de expected_elements encontrados na resposta (busca textual)
  traj_norms  — norma de mₜ ao longo da conversa de seeding
  traj_cos    — similaridade cosseno entre mₜ sucessivos (estabilidade de trajetória)

Para rodar:
  1. Certifique-se de que o Ollama está rodando: ollama serve
  2. Baixe o modelo baseline: ollama pull smollm2:1.7b
  3. poetry run python -m experiments.v1_comparison.run
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List

from src.model.loader import InstrumentedModel
from src.memory.base import MemoryState
from src.memory.updaters import EMAUpdater
from src.injection.injector import ActivationInjector
from src.baseline.full_context import FullContextBaseline
from src.eval.metrics import SAMPLE_BENCHMARK, score_fact_retention, EvalCase


MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B"
TARGET_LAYER = 12    # camada do meio (SmolLM2-1.7B tem 24 camadas)
EMA_ALPHA = 0.1
INJECT_SCALE = 0.1
MAX_NEW_TOKENS = 100
OLLAMA_MODEL = "smollm2:1.7b"


@dataclass
class ComparisonResult:
    case: EvalCase
    response_none: str
    response_mem: str
    response_full: str
    score_none: float
    score_mem: float
    score_full: float
    traj_norms: List[float] = field(default_factory=list)
    traj_cosines: List[float] = field(default_factory=list)


def seed_memory(model, injector, updater, memory, conversation):
    """
    Processa as mensagens de user da conversa prévia para acumular mₜ.
    Retorna lista de vetores capturados a cada passo (para análise de trajetória).
    """
    trajectory = []
    last_layer = model.num_layers - 1

    for msg in conversation:
        if msg["role"] != "user":
            continue
        inputs = model.tokenizer(msg["content"], return_tensors="pt").to(model.model.device)
        with torch.no_grad():
            model.model(**inputs)
        hidden = model._captured.get(last_layer)
        if hidden is None:
            continue
        e_t = hidden[0, -1, :]
        m_next = updater(memory.vector.to(e_t.device), e_t)
        memory.update(m_next.cpu())
        injector.set_memory(memory.vector.to(model.model.device))
        trajectory.append(memory.vector.clone())

    return trajectory


def trajectory_stats(vecs):
    """Retorna (cosenos, normas) dos vetores de trajetória."""
    norms = [v.norm().item() for v in vecs]
    if len(vecs) < 2:
        return [], norms
    cosines = [
        F.cosine_similarity(vecs[i - 1].unsqueeze(0), vecs[i].unsqueeze(0)).item()
        for i in range(1, len(vecs))
    ]
    return cosines, norms


def run():
    print("=" * 60)
    print("EXPERIMENTO v1: Benchmark comparativo de memória")
    print("=" * 60)

    # 1. Carregar modelo HF
    print(f"\n[1] Carregando {MODEL_NAME}...")
    model = InstrumentedModel(MODEL_NAME, device="auto")
    model.load()
    hidden_dim = model.hidden_dim
    print(f"    {model.num_layers} camadas, hidden_dim={hidden_dim}")

    # 2. Componentes reutilizáveis
    updater = EMAUpdater(alpha=EMA_ALPHA)
    injector = ActivationInjector(
        memory_dim=hidden_dim, hidden_dim=hidden_dim, scale=INJECT_SCALE
    )
    baseline = FullContextBaseline(model=OLLAMA_MODEL)

    results: List[ComparisonResult] = []

    for i, case in enumerate(SAMPLE_BENCHMARK):
        print(f"\n{'─' * 60}")
        print(f"Caso {i + 1}/{len(SAMPLE_BENCHMARK)} [{case.category}]")
        print(f"Query: {case.query}")

        # ── Condição 1: sem memória ──────────────────────────────
        model.remove_all_hooks()
        response_none = model.generate(case.query, max_new_tokens=MAX_NEW_TOKENS)
        score_none = score_fact_retention(response_none, case.expected_elements)
        print(f"\n  [sem_memoria]  score={score_none:.2f}")
        print(f"  {response_none[:120]!r}")

        # ── Condição 2: EMA + injeção em ativações ───────────────
        model.remove_all_hooks()
        memory = MemoryState(dim=hidden_dim)
        injector.set_memory(memory.vector)

        model.register_read_hook(model.num_layers - 1)
        model.register_write_hook(TARGET_LAYER, injector.modifier_fn)

        trajectory = seed_memory(model, injector, updater, memory, case.conversation)
        response_mem = model.generate(case.query, max_new_tokens=MAX_NEW_TOKENS)
        score_mem = score_fact_retention(response_mem, case.expected_elements)

        cosines, norms = trajectory_stats(trajectory)
        norm_str = f"|mₜ|={norms[-1]:.4f}" if norms else "mₜ=zero"
        print(f"\n  [ema_memoria]  score={score_mem:.2f}  {norm_str}")
        if cosines:
            print(f"  cos(mₜ,mₜ₊₁)={[f'{c:.3f}' for c in cosines]}")
        print(f"  {response_mem[:120]!r}")

        # ── Condição 3: full context via Ollama ──────────────────
        baseline.reset()
        baseline.history = list(case.conversation)
        try:
            response_full = baseline.chat(case.query)
            score_full = score_fact_retention(response_full, case.expected_elements)
        except Exception as e:
            response_full = f"[ERRO Ollama: {e}]"
            score_full = 0.0
        print(f"\n  [full_context] score={score_full:.2f}")
        print(f"  {response_full[:120]!r}")

        results.append(
            ComparisonResult(
                case=case,
                response_none=response_none,
                response_mem=response_mem,
                response_full=response_full,
                score_none=score_none,
                score_mem=score_mem,
                score_full=score_full,
                traj_norms=norms,
                traj_cosines=cosines,
            )
        )

    # ── Sumário por categoria ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMÁRIO POR CATEGORIA")
    print(f"{'=' * 60}")
    print(f"{'categoria':<22} {'sem_mem':>8} {'ema_mem':>8} {'full_ctx':>9}")
    print(f"{'─' * 22} {'─' * 8} {'─' * 8} {'─' * 9}")

    categories = sorted({r.case.category for r in results})
    for cat in categories:
        cat_results = [r for r in results if r.case.category == cat]
        n = len(cat_results)
        avg_none = sum(r.score_none for r in cat_results) / n
        avg_mem = sum(r.score_mem for r in cat_results) / n
        avg_full = sum(r.score_full for r in cat_results) / n
        print(f"{cat:<22} {avg_none:>8.2f} {avg_mem:>8.2f} {avg_full:>9.2f}")

    print(f"{'─' * 22} {'─' * 8} {'─' * 8} {'─' * 9}")
    avg_none = sum(r.score_none for r in results) / len(results)
    avg_mem = sum(r.score_mem for r in results) / len(results)
    avg_full = sum(r.score_full for r in results) / len(results)
    print(f"{'TOTAL':<22} {avg_none:>8.2f} {avg_mem:>8.2f} {avg_full:>9.2f}")

    # ── Estabilidade de trajetória ───────────────────────────────
    print(f"\n{'=' * 60}")
    print("ESTABILIDADE DE TRAJETÓRIA (EMA)")
    print(f"{'=' * 60}")
    for r in results:
        label = f"[{r.case.category}]"
        if r.traj_norms:
            print(f"  {label:<18} normas  = {[f'{n:.3f}' for n in r.traj_norms]}")
        if r.traj_cosines:
            print(f"  {'':<18} cosenos = {[f'{c:.3f}' for c in r.traj_cosines]}")

    print("\nv1 concluído.")


if __name__ == "__main__":
    run()
