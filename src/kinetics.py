# src/kinetics.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
import math

R_GAS = 8.314462618  # Дж/(моль·К)

@dataclass
class Regime:
    order: float
    A_dist: float; Ea_dist: float
    A_coke: float; Ea_coke: float

@dataclass
class VR3Kinetics:
    """Упрощённая кинетика VR→(дистилляты + кокс) с трёхрежимной зависимостью порядка."""

    # Базовые температуры переключения (°C) для VR3 из статьи
    T1_C: float = 487.8
    T2_C: float = 570.1

    # Масштабы ветвей (без калибровки)
    scale_dist: float = 0.002
    scale_coke: float = 0.50

    # Дополнительные калибруемые параметры (границы см. спецификацию)
    A_dist_scale: float = 1.0
    A_coke_scale: float = 1.0
    dT1: float = 0.0
    dT2: float = 0.0
    phi_por: float = 0.5

    # Arrhenius-пары и порядки для трёх диапазонов
    reg1:  Regime = field(default_factory=lambda: Regime(1.0, 7.8408e4, 1.1045e5, 7.8408e4, 1.1045e5))
    reg15: Regime = field(default_factory=lambda: Regime(1.5, 3.3909e18, 3.0287e5, 3.3909e17, 3.0287e5))
    reg2:  Regime = field(default_factory=lambda: Regime(2.0, 2.1660e11, 1.8316e5, 2.1660e10, 1.8316e5))

    def _regime(self, T_C: float) -> Regime:
        T1 = self.T1_C + self.dT1
        T2 = self.T2_C + self.dT2
        if T_C < T1:
            return self.reg1
        elif T_C < T2:
            return self.reg15
        return self.reg2

    def porosity_factor(self, gamma: float) -> float:
        """Линейная интерполяция между phi_por (при плотном слое) и 1 при γ≥0.4."""
        gamma_clipped = max(min(gamma, 0.4), 0.0)
        if gamma_clipped >= 0.4:
            return 1.0
        span = 0.4
        frac = gamma_clipped / span if span > 0 else 1.0
        return self.phi_por + (1.0 - self.phi_por) * frac

    def rates(self, T_C: float) -> tuple[float, float, float]:
        """Возвращает (k_dist, k_coke, order) при температуре в °C."""
        r = self._regime(T_C)
        Tk = T_C + 273.15
        k_dist = self.scale_dist * self.A_dist_scale * r.A_dist * math.exp(-r.Ea_dist / (R_GAS * Tk))
        k_coke = self.scale_coke * self.A_coke_scale * r.A_coke * math.exp(-r.Ea_coke / (R_GAS * Tk))
        return k_dist, k_coke, r.order
