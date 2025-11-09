# src/params.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from .geometry import Geometry

@dataclass
class Inlet:
    T_in_C: float = 370.0
    m_dot_kg_s: float = 5e-3 / 60.0   # 5 g/min
    rho_vr: float = 1050.0            # кг/м³ (VR3)
    v_gas_base_factor: float = 8.0    # псевдоскорость газа относительно жидкости
    p_kg_cm2: float = 5.0             # рабочее давление, кгс/см² (используется в теплообмене)

    def velocity(self, geom: Geometry) -> float:
        return self.m_dot_kg_s / (self.rho_vr * geom.A)

    def velocity_gas(self, geom: Geometry, porosity: float = 1.0) -> float:
        eff = self.v_gas_base_factor / max(porosity, 0.3)
        return self.velocity(geom) * min(eff, 25.0)

@dataclass
class Walls:
    T_wall_C: float = 510.0
    tau_heat_bottom_s: float = 2.0 * 3600.0
    tau_heat_top_s:    float = 14.0 * 3600.0   # ← было 6 ч, вернули 14 ч
    tau_profile_beta:  float = 2.5             # можешь оставить 2.8 — не критично

@dataclass
class Materials:
    rho_coke_bulk: float = 950.0   # кг/м³
    rho_dist_vap: float = 2.5      # кг/м³ (условный пар)
    porosity_min: float = 0.30     # минимальная пористость в слое
    porosity_initial: float = 1.0
    d_particle_mm: float = 1.0

@dataclass
class TimeSetup:
    total_hours: float = 12.0
    dt: float = 30.0
    snapshots_h: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    contour_every_s: float = 10.0 * 60.0

@dataclass
class WallLayer:
    """Параметры отдельного слоя стенки/футеровки."""

    k: float
    rho: float
    cp: float
    thickness: float
    epsilon: float = 0.85


@dataclass
class WallEnergy:
    """Настройки двухузловой модели стенки барабана."""

    outer: WallLayer
    inner: WallLayer
    h_amb: float
    T_amb_C: float
    zones: int = 3


@dataclass
class MixtureEnergy:
    """Эффективные параметры энергетики смеси."""

    lambda_eff: float
    cp_eff: float
    h0_mix: float
    alpha_mdot: float
    alpha_p: float
    mdot_ref: float
    p_ref: float
    dH_dist: float = 0.0
    dH_coke: float = 0.0


def defaults():
    return Geometry(), Inlet(), Walls(), Materials(), TimeSetup()
