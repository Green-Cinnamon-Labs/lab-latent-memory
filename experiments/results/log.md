# Experiment Log

Entradas em ordem cronológica inversa: o mais recente aparece primeiro.
Cada entrada deve registrar: condições exatas, resultados numéricos, observação honesta.

---

## [v1.1] Scale sweep — identidade na injeção
**Data:** 2026-04-09  
**Script:** `experiments/v1_comparison/run.py`  
**Backbone:** `HuggingFaceTB/SmolLM2-1.7B`  
**Mudança em relação à entrada anterior:** projeção substituída por identidade (memory_dim == hidden_dim); sweep de scale; métricas de trajetória expandidas  

### Condições
| parâmetro       | valor                          |
| --------------- | ------------------------------ |
| updater         | EMA α=0.1                      |
| target_layer    | 12                             |
| scales testados | 0.0, 0.01, 0.05, 0.1, 0.5, 1.0 |
| injeção         | identidade (sem projeção)      |
| max_new_tokens  | 100                            |

### Scores por scale
| scale              | fact_retention | constraint | total |
| ------------------ | -------------- | ---------- | ----- |
| 0.00 (sem_memoria) | 0.00           | 1.00       | 0.33  |
| 0.01               | 0.00           | 1.00       | 0.33  |
| 0.05               | 0.00           | 1.00       | 0.33  |
| 0.10               | 0.00           | 1.00       | 0.33  |
| 0.50               | 0.00           | 0.00       | 0.00  |
| 1.00               | 0.00           | 0.00       | 0.00  |
| full_context       | 0.50           | 1.00       | 0.67  |

### Trajetória de mₜ (scale=1.0, último seeding)
| caso              | normas (t0, t1) | cos (t1) | Δ (t1) | std (t0, t1) |
| ----------------- | --------------- | -------- | ------ | ------------ |
| fact_retention #1 | 293.66          | —        | —      | 6.48         |
| fact_retention #2 | 320.92 → 516.77 | 0.952    | 232.65 | 7.09 → 11.41 |
| constraint #1     | 323.33 → 442.60 | 0.945    | 172.89 | 7.14 → 9.78  |

### Observações

**O modelo não responde à pergunta.** `sem_memoria` retorna código Python/markdown nos casos `fact_retention` — o backbone sem prompt de sistema não segue instrução conversacional. O score 0.00 nesses casos não é "memória falhou", é "o modelo não está em modo de chat".

**A injeção até scale=0.1 não altera nada.** Scores idênticos a `sem_memoria` em todos os casos. mₜ com norma ~300-580 somado com scale=0.1 resulta em perturbação de ~30-58 sobre ativações da camada 12 — mas isso não muda a resposta em nada. Ou o sinal está sendo absorvido pelas camadas seguintes sem efeito, ou o ponto de injeção (camada 12 do meio) é insensível a essa perturbação.

**A injeção em scale≥0.5 destrói a geração.** Saída degenera em repetições e newlines. Isso confirma que mₜ com norma ~300-580 é grande demais para ser somado diretamente — a escala de ativações internas do modelo é outra ordem de grandeza.

**full_context foi o único a recuperar fato.** Score 0.50 em `fact_retention` (caso 2 passou, caso 1 falhou — o Ollama respondeu como se fosse o próprio assistente, não reconhecendo "Matheus"). Indica que o baseline funciona parcialmente.

**Conclusão desta rodada:** o experimento não pode ser avaliado ainda porque o backbone não está em modo de chat. O problema não é a memória — é a superfície de resposta. Próximo passo: usar prompt de sistema ou template de chat do SmolLM2 para que o modelo responda perguntas antes de qualquer injeção.

---

## [v1.0] Primeira rodada — projeção aleatória (baseline metodologicamente errado)
**Data:** 2026-04-07  
**Script:** `experiments/v1_comparison/run.py` (commit: feat: add v1 comparison benchmark)  
**Backbone:** `HuggingFaceTB/SmolLM2-1.7B`  
**Status:** não rodado — identificado bug metodológico antes da execução  

### Problema identificado
`ActivationInjector` usava `nn.Linear` com inicialização Kaiming (aleatória).
Injeção real: `0.1 * W_random @ mₜ` — deformação arbitrária, não memória.
Resultado de `ema_memoria` seria não-interpretável: impossível separar "memória não ajuda" de "projeção corrompeu o sinal".

### Decisão
Não rodar. Corrigir primeiro. Próxima entrada é v1.1.
