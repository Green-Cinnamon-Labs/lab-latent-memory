"""
Experimento v1: benchmark comparativo de memória com scale sweep.

Três condições:
  sem_memoria  — query sem histórico e sem injeção (scale=0)
  ema_memoria  — mₜ acumulado via EMA + injeção direta (identidade, sem projeção)
                 rodado para cada scale em SCALES
  full_context — histórico completo via Ollama

Métricas de score:
  score  — fração de expected_elements encontrados na resposta (busca textual)

Métricas de trajetória de mₜ (por turno de seeding):
  norm         — |mₜ|₂ : energia acumulada
  cos_prev     — cos(mₜ, mₜ₋₁) : estabilidade direcional
  step_size    — |mₜ - mₜ₋₁|₂ : quanto mudou no passo
  component_std — std dos componentes : quão distribuída está a informação

Para rodar:
  ollama serve                          (em terminal separado)
  ollama pull smollm2:1.7b
  poetry run python -m experiments.v1_comparison.run
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional

from src.model.loader import InstrumentedModel
from src.memory.base import MemoryState
from src.memory.updaters import EMAUpdater
from src.injection.injector import ActivationInjector
from src.baseline.full_context import FullContextBaseline
from src.eval.metrics import SAMPLE_BENCHMARK, score_fact_retention, EvalCase


MODEL_NAME   = "HuggingFaceTB/SmolLM2-1.7B"
TARGET_LAYER = 12
EMA_ALPHA    = 0.1
SCALES       = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
MAX_NEW_TOKENS = 100
OLLAMA_MODEL = "smollm2:1.7b"


# ── Métricas de trajetória ───────────────────────────────────────────────────

@dataclass
class TrajStep:
    norm: float
    cos_prev: Optional[float]   # None no primeiro passo (sem anterior)
    step_size: Optional[float]  # None no primeiro passo
    component_std: float


def compute_traj_step(current: torch.Tensor, previous: Optional[torch.Tensor]) -> TrajStep:
    norm = current.norm().item()
    std  = current.std().item()
    if previous is None:
        return TrajStep(norm=norm, cos_prev=None, step_size=None, component_std=std)
    cos = F.cosine_similarity(previous.unsqueeze(0), current.unsqueeze(0)).item()
    step = (current - previous).norm().item()
    return TrajStep(norm=norm, cos_prev=cos, step_size=step, component_std=std)


def fmt_traj(steps: List[TrajStep]) -> str:
    parts = []
    for i, s in enumerate(steps):
        cos_str  = f"cos={s.cos_prev:.3f}" if s.cos_prev is not None else "cos=—"
        step_str = f"Δ={s.step_size:.4f}" if s.step_size is not None else "Δ=—"
        parts.append(f"  t{i}: |m|={s.norm:.4f}  {cos_str}  {step_str}  std={s.component_std:.4f}")
    return "\n".join(parts)


# ── Seeding de memória ───────────────────────────────────────────────────────

def seed_memory(model, injector, updater, memory, conversation) -> List[TrajStep]:
    traj: List[TrajStep] = []
    last_layer = model.num_layers - 1
    prev_vec: Optional[torch.Tensor] = None

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

        step = compute_traj_step(memory.vector, prev_vec)
        traj.append(step)
        prev_vec = memory.vector.clone()

    return traj


# ── Runner principal ─────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("EXPERIMENTO v1.1 — scale sweep, injeção identidade")
    print("=" * 65)

    print(f"\n[1] Carregando {MODEL_NAME}...")
    model = InstrumentedModel(MODEL_NAME, device="auto")
    model.load()
    hidden_dim = model.hidden_dim
    print(f"    {model.num_layers} camadas, hidden_dim={hidden_dim}")

    updater  = EMAUpdater(alpha=EMA_ALPHA)
    injector = ActivationInjector(memory_dim=hidden_dim, hidden_dim=hidden_dim)
    baseline = FullContextBaseline(model=OLLAMA_MODEL)

    # scores[scale_str][category] → lista de floats
    scores: dict = {}
    traj_log: dict = {}  # case_label → List[TrajStep] (última rodada, scale independente)

    # ── Condição sem_memoria (scale 0.0 sem hooks) ───────────────────────────
    print("\n" + "─" * 65)
    print("Condição: sem_memoria")
    scores["sem_memoria"] = {}
    for case in SAMPLE_BENCHMARK:
        model.remove_all_hooks()
        resp = model.generate(case.query, max_new_tokens=MAX_NEW_TOKENS)
        sc   = score_fact_retention(resp, case.expected_elements)
        scores["sem_memoria"].setdefault(case.category, []).append(sc)
        print(f"  [{case.category}] score={sc:.2f}  {resp[:80]!r}")

    # ── Condição ema_memoria por scale ───────────────────────────────────────
    for scale in SCALES:
        key = f"ema_s={scale}"
        scores[key] = {}
        print(f"\n{'─' * 65}")
        print(f"Condição: ema_memoria  scale={scale}")

        for case in SAMPLE_BENCHMARK:
            model.remove_all_hooks()
            memory = MemoryState(dim=hidden_dim)
            injector.set_scale(scale)
            injector.set_memory(memory.vector)

            model.register_read_hook(model.num_layers - 1)
            model.register_write_hook(TARGET_LAYER, injector.modifier_fn)

            traj = seed_memory(model, injector, updater, memory, case.conversation)

            resp = model.generate(case.query, max_new_tokens=MAX_NEW_TOKENS)
            sc   = score_fact_retention(resp, case.expected_elements)
            scores[key].setdefault(case.category, []).append(sc)

            label = f"{case.category}_{len(scores[key][case.category])}"
            traj_log[label] = traj  # sobrescreve a cada scale; usado na última iteração

            print(f"  [{case.category}] score={sc:.2f}  {resp[:80]!r}")
            if traj:
                last = traj[-1]
                cos_str = f"{last.cos_prev:.3f}" if last.cos_prev is not None else "—"
                print(f"    |mₜ|={last.norm:.4f}  cos={cos_str}  std={last.component_std:.4f}")

    # ── Condição full_context ────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("Condição: full_context (Ollama)")
    scores["full_context"] = {}
    for case in SAMPLE_BENCHMARK:
        baseline.reset()
        baseline.history = list(case.conversation)
        try:
            resp = baseline.chat(case.query)
            sc   = score_fact_retention(resp, case.expected_elements)
        except Exception as e:
            resp = f"[ERRO: {e}]"
            sc   = 0.0
        scores["full_context"].setdefault(case.category, []).append(sc)
        print(f"  [{case.category}] score={sc:.2f}  {resp[:80]!r}")

    # ── Tabela de scores ─────────────────────────────────────────────────────
    categories = sorted({c for v in scores.values() for c in v})
    col_w = 10

    print(f"\n{'=' * 65}")
    print("SCORES POR SCALE E CATEGORIA")
    print(f"{'=' * 65}")
    header = f"{'condição':<22}" + "".join(f"{c:>{col_w}}" for c in categories) + f"{'TOTAL':>{col_w}}"
    print(header)
    print("─" * len(header))

    for cond, cat_scores in scores.items():
        all_vals = [v for vals in cat_scores.values() for v in vals]
        total = sum(all_vals) / len(all_vals) if all_vals else 0.0
        row = f"{cond:<22}"
        for cat in categories:
            vals = cat_scores.get(cat, [])
            avg = sum(vals) / len(vals) if vals else 0.0
            row += f"{avg:>{col_w}.2f}"
        row += f"{total:>{col_w}.2f}"
        print(row)

    # ── Trajetória de mₜ ─────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"TRAJETÓRIA DE mₜ (scale={SCALES[-1]}, último seeding por caso)")
    print(f"{'=' * 65}")
    for label, traj in traj_log.items():
        print(f"\n  {label}:")
        print(fmt_traj(traj))

    print("\nv1.1 concluído. Preencher experiments/results/log.md com os valores acima.")


if __name__ == "__main__":
    run()
