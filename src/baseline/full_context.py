"""
Baseline: LLM com reinjeção integral do histórico.

Este é o padrão contra o qual você compara.
Usa Ollama como servidor local, enviando toda a conversa a cada turno.

Se yₜ_full ≈ yₜ_mem, seu objeto mₜ está funcionando.
"""

import requests
from typing import List, Dict


class FullContextBaseline:
    """
    Envia o histórico completo para o modelo via Ollama a cada turno.
    """

    def __init__(self, model: str = "phi", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.history: List[Dict[str, str]] = []

    def chat(self, user_message: str) -> str:
        """
        Envia mensagem com histórico completo. Retorna resposta.
        """
        self.history.append({"role": "user", "content": user_message})

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": self.history,
                "stream": False,
            },
        )
        response.raise_for_status()
        assistant_msg = response.json()["message"]["content"]

        self.history.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg

    def reset(self):
        """Limpa o histórico."""
        self.history.clear()

    @property
    def total_tokens_sent(self) -> int:
        """Estimativa grosseira de tokens reinjetados (para medir custo)."""
        return sum(len(m["content"].split()) for m in self.history)
