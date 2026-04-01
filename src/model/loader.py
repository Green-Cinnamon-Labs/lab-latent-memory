"""
Carregamento do modelo HuggingFace com hooks para interceptação.

Este módulo te dá acesso direto às ativações internas do transformer.
Cada camada do modelo produz um tensor intermediário (hidden state).
Um "hook" é uma função que o PyTorch chama automaticamente quando
aquela camada executa. Com isso você pode:
  - ler o que a camada produziu
  - modificar a saída antes dela seguir para a próxima camada

Isso é a base da Porta 3: injetar mₜ diretamente na computação.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class InstrumentedModel:
    """
    Wrapper que carrega um modelo HF e permite registrar hooks
    em camadas específicas para leitura e modificação de ativações.
    """

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._hooks = []
        self._captured = {}  # {layer_idx: tensor} capturado pelos hooks

    def load(self):
        """Carrega modelo e tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=self.device,
        )
        self.model.eval()
        return self

    def get_layers(self):
        """
        Retorna a lista de camadas (transformer blocks) do modelo.
        A estrutura varia por modelo, mas geralmente é algo como:
          model.model.layers       (LLaMA, Phi, Mistral)
          model.transformer.h      (GPT-2)
        """
        # Tenta caminhos comuns
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError(
            f"Não consegui encontrar as camadas do modelo {self.model_name}. "
            "Inspecione model.named_modules() para descobrir a estrutura."
        )

    def register_read_hook(self, layer_idx: int):
        """
        Registra um hook que CAPTURA a saída de uma camada.
        Isso te permite ver o que o modelo está computando internamente.

        O tensor capturado fica em self._captured[layer_idx].
        """
        layers = self.get_layers()
        layer = layers[layer_idx]

        def hook_fn(module, input, output):
            # output pode ser tuple (hidden_states, ...) dependendo do modelo
            if isinstance(output, tuple):
                self._captured[layer_idx] = output[0].detach()
            else:
                self._captured[layer_idx] = output.detach()

        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle

    def register_write_hook(self, layer_idx: int, modifier_fn):
        """
        Registra um hook que MODIFICA a saída de uma camada.
        
        modifier_fn: callable(hidden_states: Tensor) -> Tensor
            Recebe o tensor de ativações e retorna o tensor modificado.
            É aqui que mₜ entra na computação.

        Exemplo:
            def inject_memory(hidden_states):
                # hidden_states shape: (batch, seq_len, hidden_dim)
                # Somar mₜ projetado ao último token
                hidden_states[:, -1, :] += projected_memory
                return hidden_states
            
            model.register_write_hook(layer_idx=16, modifier_fn=inject_memory)
        """
        layers = self.get_layers()
        layer = layers[layer_idx]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                modified = modifier_fn(output[0])
                return (modified,) + output[1:]
            else:
                return modifier_fn(output)

        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle

    def remove_all_hooks(self):
        """Remove todos os hooks registrados."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    def generate(self, text: str, max_new_tokens: int = 128, **kwargs):
        """Gera resposta a partir de texto. Hooks ativos são executados."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def hidden_dim(self) -> int:
        """Dimensão dos hidden states do modelo."""
        return self.model.config.hidden_size

    @property
    def num_layers(self) -> int:
        """Número de camadas do transformer."""
        return len(self.get_layers())
