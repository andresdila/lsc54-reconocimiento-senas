"""Clasificadores secuenciales sobre landmarks (Entrega 2).

Flujo (arquitectura candidata del plan, sección 9):

    x (B, T=32, 101)
      -> proyección por cuadro: Linear(101 -> d) + LayerNorm + GELU + Dropout
           Lleva coordenadas y máscara a un espacio común; LayerNorm estabiliza la escala
           antes de la recurrencia (las coordenadas tienen colas largas aun estandarizadas).
      -> codificador temporal (uno de):
           * "gru"  : GRU bidireccional de `layers` capas y `hidden` unidades por dirección
           * "lstm" : igual con LSTM (variante del plan)
           * "conv" : convoluciones temporales 1D de tamaño comparable (ablación: ¿aporta la recurrencia?)
           La bidireccionalidad es válida porque la seña se clasifica completa (no hay
           predicción en línea).
      -> agrupamiento por atención temporal: un peso por cuadro (softmax sobre T) y suma
           ponderada. Deja que la red se concentre en el núcleo de la seña y no en las
           manos en reposo del inicio y el final.
      -> cabeza: Dropout + Linear(2*hidden [+1] -> 43)
           El +1 opcional es log(n_frames) estandarizado (la ablación "sin n_frames" lo apaga).
"""
from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)  # (B, T)
        return torch.einsum("bt,btd->bd", w, h), w


class ConvEncoder(nn.Module):
    def __init__(self, d: int, hidden: int, layers: int, dropout: float, kernel: int = 5):
        super().__init__()
        blocks, c_in = [], d
        for _ in range(layers + 1):
            blocks += [nn.Conv1d(c_in, 2 * hidden, kernel, padding=kernel // 2), nn.BatchNorm1d(2 * hidden),
                       nn.GELU(), nn.Dropout(dropout)]
            c_in = 2 * hidden
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class SeqClassifier(nn.Module):
    def __init__(self, d_in: int, n_classes: int, encoder: str = "gru", d_model: int = 128, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3, rnn_dropout: float = 0.1, use_n_frames: bool = True):
        super().__init__()
        self.encoder_type, self.use_n_frames = encoder, use_n_frames
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(rnn_dropout))
        if encoder in ("gru", "lstm"):
            rnn = nn.GRU if encoder == "gru" else nn.LSTM
            self.encoder = rnn(d_model, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                               dropout=rnn_dropout if layers > 1 else 0.0)
        elif encoder == "conv":
            self.encoder = ConvEncoder(d_model, hidden, layers, rnn_dropout)
        else:
            raise ValueError(encoder)
        self.pool = AttentionPool(2 * hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden + int(use_n_frames), n_classes))

    def forward(self, x: torch.Tensor, n_frames: torch.Tensor | None = None, return_attention: bool = False):
        h = self.proj(x)
        h = self.encoder(h)[0] if self.encoder_type in ("gru", "lstm") else self.encoder(h)
        z, w = self.pool(h)
        if self.use_n_frames:
            z = torch.cat([z, n_frames.view(-1, 1)], dim=1)
        logits = self.head(z)
        return (logits, w) if return_attention else logits


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
