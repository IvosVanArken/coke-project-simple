# src/solver_1d.py
# -*- coding: utf-8 -*-
"""One-dimensional delayed coking solver with wall + mixture energy balances."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .geometry import Geometry
from .params import (
    Inlet,
    Walls,
    Materials,
    TimeSetup,
    WallEnergy,
    WallLayer,
    MixtureEnergy,
)
from .kinetics import VR3Kinetics


SIGMA = 5.670374419e-8  # Stefan–Boltzmann constant, W/(m²·K⁴)
TINY = 1e-12
T_COKE_ONSET_C = 415.0


def _default_wall_energy() -> WallEnergy:
    """Conservative default wall configuration for backward compatibility."""

    outer = WallLayer(k=18.0, rho=7800.0, cp=600.0, thickness=0.06, epsilon=0.85)
    inner = WallLayer(k=2.0, rho=3200.0, cp=800.0, thickness=0.10, epsilon=0.75)
    return WallEnergy(outer=outer, inner=inner, h_amb=20.0, T_amb_C=25.0, zones=3)


def _default_mixture_energy(inlet: Inlet) -> MixtureEnergy:
    return MixtureEnergy(
        lambda_eff=0.6,
        cp_eff=2200.0,
        h0_mix=120.0,
        alpha_mdot=0.5,
        alpha_p=0.1,
        mdot_ref=max(inlet.m_dot_kg_s, 1e-3),
        p_ref=2.0,
    )


def h_mix(mdot_kg_s: float, p: float, cfg: MixtureEnergy) -> float:
    """Convective heat transfer coefficient between wall and mixture."""

    mdot = max(mdot_kg_s, 1e-6)
    pref = max(cfg.mdot_ref, 1e-6)
    p_val = max(p, 1e-6)
    p_ref = max(cfg.p_ref, 1e-6)
    return cfg.h0_mix * (mdot / pref) ** cfg.alpha_mdot * (p_val / p_ref) ** cfg.alpha_p


@dataclass
class EnergyState:
    """Helper container for wall thermal masses per zone."""

    C_out: np.ndarray
    C_in: np.ndarray
    A_zone: np.ndarray
    K_wall: np.ndarray


class Coking1DSolver:
    """Finite-volume 1D solver with mass + energy balances."""

    def __init__(
        self,
        geom: Geometry,
        inlet: Inlet,
        walls: Walls,
        mats: Materials,
        tcfg: TimeSetup,
        kin: VR3Kinetics,
        wall_energy: Optional[WallEnergy] = None,
        mix_energy: Optional[MixtureEnergy] = None,
    ) -> None:
        self.g = geom
        self.inlet = inlet
        self.walls = walls
        self.mats = mats
        self.tcfg = tcfg
        self.kin = kin

        self.wall_energy = wall_energy or _default_wall_energy()
        self.mix_energy = mix_energy or _default_mixture_energy(inlet)

        self.time_s = 0.0
        self.current_pressure = 2.0  # кгс/см² (может быть обновлено снаружи)

        NZ = geom.NZ
        self.aR = np.ones(NZ, dtype=np.float64)
        self.aC = np.zeros(NZ, dtype=np.float64)
        self.aD = np.zeros(NZ, dtype=np.float64)
        self.gamma = 1.0 - self.aC

        self.porosity_min = float(getattr(self.mats, "porosity_min", 0.3))

        # Температура смеси хранится в Кельвинах, а self.T — в °C для совместимости
        T0_K = inlet.T_in_C + 273.15
        self.T_mix_K = np.full(NZ, T0_K, dtype=np.float64)
        self.T = self.T_mix_K - 273.15

        self.vol_cell = self.g.A * self.g.dz

        # Служебные массивы и предрасчётная геометрия
        self.zone_index = self._build_zone_index(self.wall_energy.zones)
        self.zone_areas = self._zone_areas(self.wall_energy.zones)
        self.energy_state = self._make_energy_state()
        self.cell_wall_area = self._cell_wall_area()

        self.T_shell_out_K = np.full(self.wall_energy.zones, T0_K, dtype=np.float64)
        self.T_shell_in_K = np.full(self.wall_energy.zones, T0_K, dtype=np.float64)
        self._shell_obs_K = [math.nan] * self.wall_energy.zones

        self.T_amb_K = self.wall_energy.T_amb_C + 273.15
        self.T_sky_K = max(self.T_amb_K - 10.0, 50.0)

        # Инвентарь VR для массового баланса
        self.m_vr0 = float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

        # Истории (оставляем совместимыми с прежней версией)
        self.snap_times_h = list(tcfg.snapshots_h)
        self._snap_idx = 0
        self.snapshots = {"t_h": [], "T": [], "aR": [], "aD": [], "aC": []}
        self.time_h_hist = []
        self.bed_eq_cm_hist = []
        self.bed_front_cm_hist = []
        self.contour_every_s = float(tcfg.contour_every_s)
        self.contour_t: list[float] = []
        self.contour_T: list[np.ndarray] = []
        self.contour_aR: list[np.ndarray] = []
        self.contour_aD: list[np.ndarray] = []
        self.contour_aC: list[np.ndarray] = []

        self.porosity_avg = 1.0
        self.k_mix_gas = 0.05

        self._take_snapshot()
        self._maybe_take_contour(force=True)
        if self.snap_times_h and abs(self.snapshots["t_h"][-1] - self.snap_times_h[0]) < 1e-9:
            self._snap_idx = 1

    # ------------------------------------------------------------------
    # Конфигурационные вспомогательные методы
    def _build_zone_index(self, zones: int) -> np.ndarray:
        if zones != 3:
            raise ValueError("Текущая реализация поддерживает ровно 3 зоны по высоте")

        NZ = self.g.NZ
        dz = self.g.dz
        z_centers = (np.arange(NZ, dtype=float) + 0.5) * dz

        head_height = min(max(self.g.D / 2.0, 0.1 * self.g.H), self.g.H / 2.0)
        if 2 * head_height > self.g.H:
            head_height = self.g.H / 2.0

        zone_idx = np.ones(NZ, dtype=np.int64)
        zone_idx[z_centers <= head_height] = 0
        zone_idx[z_centers >= self.g.H - head_height] = 2

        # Гарантируем хотя бы по одной ячейке в каждой зоне
        if not np.any(zone_idx == 0):
            zone_idx[0] = 0
        if not np.any(zone_idx == 2):
            zone_idx[-1] = 2

        return zone_idx

    def _zone_areas(self, zones: int) -> np.ndarray:
        radius = self.g.D / 2.0
        head_area = math.pi * radius ** 2
        head_height = min(max(self.g.D / 2.0, 0.1 * self.g.H), self.g.H / 2.0)
        cyl_height = max(self.g.H - 2.0 * head_height, 1e-6)
        cyl_area = 2.0 * math.pi * radius * cyl_height
        return np.array([head_area, cyl_area, head_area], dtype=float)

    def _make_energy_state(self) -> EnergyState:
        outer = self.wall_energy.outer
        inner = self.wall_energy.inner

        R_per_area = outer.thickness / max(outer.k, TINY) + inner.thickness / max(inner.k, TINY)
        K_per_area = 1.0 / max(R_per_area, TINY)

        C_out = outer.rho * outer.cp * outer.thickness * self.zone_areas
        C_in = inner.rho * inner.cp * inner.thickness * self.zone_areas
        K_wall = K_per_area * self.zone_areas
        return EnergyState(C_out=C_out, C_in=C_in, A_zone=self.zone_areas, K_wall=K_wall)

    def _cell_wall_area(self) -> np.ndarray:
        area = np.zeros(self.g.NZ, dtype=float)
        for zone in range(self.wall_energy.zones):
            mask = np.where(self.zone_index == zone)[0]
            if mask.size == 0:
                continue
            share = self.zone_areas[zone] / mask.size
            area[mask] = share
        return area

    # ------------------------------------------------------------------
    # Доступ к состоянию слоя / общие сервисы
    def bed_height_front(self, thr: float = 0.05) -> float:
        idx = np.where(self.aC > thr)[0]
        if idx.size == 0:
            return 0.0
        top_idx = idx[-1] + 1
        return float(min(top_idx * self.g.dz, self.g.H))

    def bed_height_equiv(self) -> float:
        eps_max = 1.0 - self.porosity_min
        if eps_max <= TINY:
            return 0.0
        aC_clip = np.clip(self.aC, 0.0, eps_max)
        return float(min(np.sum(aC_clip / eps_max) * self.g.dz, self.g.H))

    def coke_mass(self) -> float:
        return float(np.sum(self.mats.rho_coke_bulk * self.aC) * self.vol_cell)

    def inlet_mass_total(self) -> float:
        return float(self.inlet.m_dot_kg_s * self.time_s)

    def vr_inventory_mass(self) -> float:
        return float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

    def get_porosity_profile(self) -> np.ndarray:
        return np.maximum(self.gamma, self.porosity_min)

    def update_porosity_effects(self) -> None:
        zone = self.aC > 0.01
        if np.any(zone):
            self.porosity_avg = max(float(np.mean(self.gamma[zone])), self.porosity_min)
        else:
            self.porosity_avg = 1.0

    def coke_yield_pct_feed(self) -> float:
        return 100.0 * self.coke_mass() / max(self.inlet_mass_total(), 1e-9)

    def coke_yield_pct_balance(self) -> float:
        return 100.0 * self.coke_mass() / max(self.m_vr0 + self.inlet_mass_total(), 1e-9)

    # ------------------------------------------------------------------
    def _take_snapshot(self) -> None:
        t_h = self.time_s / 3600.0
        self.snapshots["t_h"].append(t_h)
        self.snapshots["T"].append((self.T_mix_K - 273.15).copy())
        self.snapshots["aR"].append(self.aR.copy())
        self.snapshots["aD"].append(self.aD.copy())
        self.snapshots["aC"].append(self.aC.copy())

    def _maybe_take_snapshot(self) -> None:
        if self._snap_idx >= len(self.snap_times_h):
            return
        t_h = self.time_s / 3600.0
        if t_h + 1e-4 >= self.snap_times_h[self._snap_idx]:
            self._take_snapshot()
            self._snap_idx += 1

    def _maybe_take_contour(self, force: bool = False) -> None:
        if force or (len(self.contour_t) == 0) or (
            self.time_s - self.contour_t[-1] >= self.contour_every_s - 1e-12
        ):
            self.contour_t.append(self.time_s)
            self.contour_T.append((self.T_mix_K - 273.15).copy())
            self.contour_aR.append(self.aR.copy())
            self.contour_aD.append(self.aD.copy())
            self.contour_aC.append(self.aC.copy())

    # ------------------------------------------------------------------
    # Наблюдения по оболочке
    def set_shell_observations(
        self, t: float, T_upper_C: Optional[float] = None, T_lower_C: Optional[float] = None
    ) -> None:
        if T_upper_C is not None:
            self._shell_obs_K[2] = T_upper_C + 273.15
        else:
            self._shell_obs_K[2] = math.nan

        if T_lower_C is not None:
            self._shell_obs_K[0] = T_lower_C + 273.15
        else:
            self._shell_obs_K[0] = math.nan

    # ------------------------------------------------------------------
    # Основные численные блоки
    def _advect(self, dt: float, v_liq: float, v_gas: float) -> None:
        dz = self.g.dz
        gamma = self.gamma

        sigR = max(0.0, v_liq) * dt / max(dz, TINY)
        sigD = max(0.0, v_gas) * dt / max(dz, TINY)
        nsub = int(max(1.0, math.ceil(max(sigR, sigD, 1.0))))

        for _ in range(nsub):
            dt_sub = dt / nsub
            sR = max(0.0, v_liq) * dt_sub / max(dz, TINY)

            porous = gamma < 0.95
            if np.any(porous):
                avg_por = max(float(np.mean(gamma[porous])), self.porosity_min)
            else:
                avg_por = 1.0
            sD = max(0.0, v_gas / avg_por) * dt_sub / max(dz, TINY)

            aR0 = self.aR.copy()
            aD0 = self.aD.copy()
            prevR = gamma[0]
            prevD = 0.0
            for k in range(self.g.NZ):
                curR = aR0[k]
                curD = aD0[k]
                self.aR[k] = curR - sR * (curR - prevR)
                self.aD[k] = curD - sD * (curD - prevD)
                self.aR[k] = min(max(self.aR[k], 0.0), gamma[k])
                self.aD[k] = max(self.aD[k], 0.0)
                prevR = curR
                prevD = curD

    def _react(self, dt: float) -> np.ndarray:
        q_react = np.zeros(self.g.NZ, dtype=float)
        rho_vr = self.inlet.rho_vr
        rho_coke = self.mats.rho_coke_bulk
        rho_dist = max(self.mats.rho_dist_vap, TINY)
        max_coke = 1.0 - self.porosity_min

        for k in range(self.g.NZ):
            T_C = self.T_mix_K[k] - 273.15
            if T_C < T_COKE_ONSET_C:
                continue

            gamma = self.gamma[k]
            if gamma <= self.porosity_min + 1e-6:
                continue

            aR = self.aR[k]
            if aR <= 1e-9:
                continue

            k_dist, k_coke, order = self.kin.rates(T_C)
            factor = self.kin.porosity_factor(gamma)
            k_dist *= factor
            k_coke *= factor
            k_total = k_dist + k_coke
            if k_total <= 0.0:
                continue

            rate_factor = gamma * (aR ** order)
            raw_dR = rate_factor * k_total * dt
            if raw_dR <= 0.0:
                continue

            ratio = 1.0
            if raw_dR > aR:
                ratio = aR / raw_dR

            k_dist_eff = k_dist * ratio
            k_coke_eff = k_coke * ratio
            rate_dist = rho_vr * rate_factor * k_dist_eff
            rate_coke = rho_vr * rate_factor * k_coke_eff

            dR = (rate_dist + rate_coke) * dt / max(rho_vr * gamma, TINY)
            dR = min(dR, self.aR[k])
            self.aR[k] = max(self.aR[k] - dR, 0.0)

            dC = rate_coke * dt / max(rho_coke, TINY)
            self.aC[k] = min(self.aC[k] + dC, max_coke)
            self.gamma[k] = max(1.0 - self.aC[k], self.porosity_min)

            dD = rate_dist * dt / rho_dist
            maxD = max(self.gamma[k] - self.aR[k], 0.0)
            self.aD[k] = min(self.aD[k] + dD, maxD)

            q_react[k] = (
                -self.kin.dH_dist_J_kg * rate_dist + -self.kin.dH_coke_J_kg * rate_coke
            )

        return q_react

    def _update_wall_energy(self, dt: float, h_mix_val: float) -> None:
        for zone in range(self.wall_energy.zones):
            C_out = max(self.energy_state.C_out[zone], TINY)
            C_in = max(self.energy_state.C_in[zone], TINY)
            K_wall = self.energy_state.K_wall[zone]
            A_zone = self.energy_state.A_zone[zone]

            T_out = self.T_shell_out_K[zone]
            T_in = self.T_shell_in_K[zone]

            mix_mask = self.zone_index == zone
            if np.any(mix_mask):
                T_mix_zone = float(np.mean(self.T_mix_K[mix_mask]))
            else:
                T_mix_zone = float(np.mean(self.T_mix_K))

            dT_out_dt = (
                -K_wall * (T_out - T_in)
                - self.wall_energy.h_amb * A_zone * (T_out - self.T_amb_K)
                - SIGMA * self.wall_energy.outer.epsilon * A_zone * (T_out ** 4 - self.T_sky_K ** 4)
            ) / C_out

            dT_in_dt = (
                K_wall * (T_out - T_in)
                - h_mix_val * A_zone * (T_in - T_mix_zone)
            ) / C_in

            self.T_shell_out_K[zone] = T_out + dt * dT_out_dt
            self.T_shell_in_K[zone] = T_in + dt * dT_in_dt

            obs = self._shell_obs_K[zone]
            if np.isfinite(obs):
                self.T_shell_out_K[zone] = obs

    def _update_mixture_energy(self, dt: float, h_mix_val: float, q_react: np.ndarray) -> None:
        rho = max(self.inlet.rho_vr, 1.0)
        cp = max(self.mix_energy.cp_eff, 1.0)
        rho_cp = rho * cp
        dz = self.g.dz
        T_new = self.T_mix_K.copy()

        T_inlet = self.inlet.T_in_C + 273.15
        v_liq = self.inlet.velocity(self.g) + self.k_mix_gas * self.inlet.velocity_gas(self.g, self.porosity_avg)

        for k in range(self.g.NZ):
            T_k = self.T_mix_K[k]
            T_prev = T_inlet if k == 0 else self.T_mix_K[k - 1]
            T_next = self.T_mix_K[k + 1] if k + 1 < self.g.NZ else self.T_shell_in_K[self.zone_index[k]]

            dTdz = (T_k - T_prev) / max(dz, TINY)
            d2Tdz2 = (T_next - 2.0 * T_k + T_prev) / max(dz ** 2, TINY)

            cond_term = self.mix_energy.lambda_eff * d2Tdz2 / rho_cp
            adv_term = -v_liq * dTdz
            ht_term = (
                h_mix_val
                * (self.cell_wall_area[k] / max(self.vol_cell, TINY))
                * (self.T_shell_in_K[self.zone_index[k]] - T_k)
                / rho_cp
            )
            react_term = q_react[k] / rho_cp

            dTdt = adv_term + cond_term + ht_term + react_term
            T_new[k] = T_k + dt * dTdt

        self.T_mix_K = T_new
        self.T = self.T_mix_K - 273.15

    # ------------------------------------------------------------------
    def step(self, pressure: Optional[float] = None) -> None:
        self.update_porosity_effects()

        if pressure is not None:
            self.current_pressure = pressure

        v_liq = self.inlet.velocity(self.g) + self.k_mix_gas * self.inlet.velocity_gas(
            self.g, self.porosity_avg
        )
        v_gas = self.inlet.velocity_gas(self.g, self.porosity_avg)

        dt = self.tcfg.dt
        self._advect(dt, v_liq, v_gas)
        q_react = self._react(dt)

        h_mix_val = h_mix(self.inlet.m_dot_kg_s, self.current_pressure, self.mix_energy)
        self._update_wall_energy(dt, h_mix_val)
        self._update_mixture_energy(dt, h_mix_val, q_react)

        self.time_s += dt

    # ------------------------------------------------------------------
    def run(self, verbose_hourly: bool = True) -> dict:
        steps = int(self.tcfg.total_hours * 3600.0 / self.tcfg.dt)
        next_hour_s = 3600.0

        for _ in range(steps):
            self.step()

            if self.time_s + 1e-9 >= next_hour_s:
                self.time_h_hist.append(self.time_s / 3600.0)
                self.bed_eq_cm_hist.append(self.bed_height_equiv() * 100.0)
                self.bed_front_cm_hist.append(self.bed_height_front() * 100.0)
                if verbose_hourly:
                    y_feed = self.coke_yield_pct_feed()
                    y_bal = self.coke_yield_pct_balance()
                    if self.time_s < 2 * 3600.0:
                        print(
                            f"t = {self.time_s/3600.0:5.1f} ч | H_eq={self.bed_eq_cm_hist[-1]:.1f} см | "
                            f"H_front={self.bed_front_cm_hist[-1]:.1f} см | Yбал={y_bal:5.2f}% | "
                            f"T_avg={np.mean(self.T):.1f}°C | ε_avg={self.porosity_avg:.3f}"
                        )
                    else:
                        print(
                            f"t = {self.time_s/3600.0:5.1f} ч | H_eq={self.bed_eq_cm_hist[-1]:.1f} см | "
                            f"H_front={self.bed_front_cm_hist[-1]:.1f} см | Yfeed={y_feed:5.2f}% | "
                            f"Yбал={y_bal:5.2f}% | T_avg={np.mean(self.T):.1f}°C | ε_avg={self.porosity_avg:.3f}"
                        )
                next_hour_s += 3600.0

            self._maybe_take_snapshot()
            self._maybe_take_contour()

        results = {
            "H_bed_m": self.bed_height_equiv(),
            "H_front_m": self.bed_height_front(),
            "yield_pct": self.coke_yield_pct_feed(),
            "T_avg_C": float(np.mean(self.T)),
            "porosity_avg": float(self.porosity_avg),
            "final": {
                "T": (self.T_mix_K - 273.15).copy(),
                "aR": self.aR.copy(),
                "aD": self.aD.copy(),
                "aC": self.aC.copy(),
            },
            "z": self.g.z.copy(),
            "snapshots": {
                "t_h": np.array(self.snapshots["t_h"], dtype=float),
                "T": np.stack(self.snapshots["T"], axis=0) if self.snapshots["T"] else np.zeros((0, self.g.NZ)),
                "aR": np.stack(self.snapshots["aR"], axis=0) if self.snapshots["aR"] else np.zeros((0, self.g.NZ)),
                "aD": np.stack(self.snapshots["aD"], axis=0) if self.snapshots["aD"] else np.zeros((0, self.g.NZ)),
                "aC": np.stack(self.snapshots["aC"], axis=0) if self.snapshots["aC"] else np.zeros((0, self.g.NZ)),
            },
            "growth": {
                "t_h": np.array(self.time_h_hist, dtype=float),
                "H_cm": np.array(self.bed_eq_cm_hist, dtype=float),
                "H_front_cm": np.array(self.bed_front_cm_hist, dtype=float),
            },
            "contours": {
                "t_s": np.array(self.contour_t, dtype=float),
                "T": np.stack(self.contour_T, axis=0) if self.contour_T else np.zeros((0, self.g.NZ)),
                "aR": np.stack(self.contour_aR, axis=0) if self.contour_aR else np.zeros((0, self.g.NZ)),
                "aD": np.stack(self.contour_aD, axis=0) if self.contour_aD else np.zeros((0, self.g.NZ)),
                "aC": np.stack(self.contour_aC, axis=0) if self.contour_aC else np.zeros((0, self.g.NZ)),
            },
            "meta": {
                "A_m2": self.g.A,
                "m_dot_kg_s": self.inlet.m_dot_kg_s,
                "rho_vr": self.inlet.rho_vr,
                "rho_coke": self.mats.rho_coke_bulk,
                "m_vr0": self.m_vr0,
            },
            "thermal": {
                "T_shell_out_C": self.T_shell_out_K - 273.15,
                "T_shell_in_C": self.T_shell_in_K - 273.15,
            },
        }
        return results

