# src/solver_1d.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math, numpy as np
from typing import Optional

try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False

from .geometry import Geometry
from .params import Inlet, Walls, Materials, TimeSetup, WallEnergy, MixtureEnergy
from .kinetics import VR3Kinetics

# Порог начала коксования (ниже реакции выключены)
T_COKE_ONSET_C = 415.0
SIGMA_SB = 5.670374419e-8  # постоянная Стефана–Больцмана, W/(m²·K⁴)


def h_mix(mdot_kg_s: float, p: float, h0_mix: float, alpha_mdot: float,
          alpha_p: float, mdot_ref: float, p_ref: float) -> float:
    """Коэффициент теплоотдачи стенка→смесь (W/(m²·K))."""
    md = max(float(mdot_kg_s), 1e-6)
    md_ref = max(float(mdot_ref), 1e-6)
    pr = max(float(p), 1e-6)
    pr_ref = max(float(p_ref), 1e-6)
    return float(h0_mix) * (md / md_ref) ** float(alpha_mdot) * (pr / pr_ref) ** float(alpha_p)

def _make_tau_profile(geom: Geometry, walls: Walls) -> np.ndarray:
    tau_bottom = getattr(walls, "tau_heat_bottom_s", 2.0*3600.0)
    tau_top    = getattr(walls, "tau_heat_top_s",    6.0*3600.0)
    beta = float(getattr(walls, "tau_profile_beta", 2.0))
    z = geom.z
    s = (z/geom.H)**beta if geom.H > 0 else np.zeros_like(z)
    return (tau_bottom + (tau_top - tau_bottom) * s).astype(np.float64)

if NUMBA:
    @njit(cache=True, fastmath=True)
    def _nb_step_advect_react(T, aR, aC, aD, gamma,
                              dt, rho_vr, rho_coke, rho_dist,
                              T_wall, tau_z, vR, vD, dz,
                              T1_C, T2_C, T_onset_C,
                              A1d, E1d, A1c, E1c, o1,
                              A15d, E15d, A15c, E15c, o15,
                              A2d, E2d, A2c, E2c, o2,
                              scale_dist, scale_coke,
                              porosity_min, phi_por):
        R = 8.314462618
        NZ = T.shape[0]

        # 1) прогрев
        for k in range(NZ):
            T[k] += (T_wall - T[k]) * (dt / tau_z[k])

        # 2) адвекция (upwind, с подсубшагами)
        sigR = max(0.0, vR) * dt / dz
        sigD = max(0.0, vD) * dt / dz
        nsub = 1
        if sigR > 1.0 or sigD > 1.0:
            nsub = int(math.ceil(max(sigR, sigD)))

        for _ in range(nsub):
            dt_sub = dt / nsub
            sR = max(0.0, vR) * dt_sub / dz

            # усиление газовой скорости при низкой пористости
            sum_por = 0.0;
            n = 0
            for k in range(NZ):
                if gamma[k] < 0.95:
                    sum_por += gamma[k];
                    n += 1
            avg_por = max((sum_por / n) if n > 0 else 1.0, porosity_min)
            sD = max(0.0, vD / avg_por) * dt_sub / dz

            aR_in, aD_in = gamma[0], 0.0
            aR0 = aR.copy(); aD0 = aD.copy()
            prevR, prevD = aR_in, aD_in
            for k in range(NZ):
                curR, curD = aR0[k], aD0[k]
                aR[k] = curR - sR * (curR - prevR)
                aD[k] = curD - sD * (curD - prevD)
                if aR[k] < 0.0: aR[k] = 0.0
                if aR[k] > gamma[k]: aR[k] = gamma[k]
                if aD[k] < 0.0: aD[k] = 0.0
                prevR, prevD = curR, curD

        # 3) реакции (с замедлением в плотном коксе)
        for k in range(NZ):
            Tc = T[k]
            if Tc < T_onset_C:
                continue

            if Tc < T1_C:
                order, Ad, Ed, Ac, Ec = o1, A1d, E1d, A1c, E1c
            elif Tc < T2_C:
                order, Ad, Ed, Ac, Ec = o15, A15d, E15d, A15c, E15c
            else:
                order, Ad, Ed, Ac, Ec = o2, A2d, E2d, A2c, E2c

            Tk = Tc + 273.15
            k_dist = scale_dist * Ad * math.exp(-Ed / (R * Tk))
            k_coke = scale_coke * Ac * math.exp(-Ec / (R * Tk))

            g = gamma[k]; r0 = aR[k]

            if g < 0.4:
                factor = phi_por + (1.0 - phi_por) * (g/0.4)
                k_dist *= factor; k_coke *= factor

            k_tot = k_dist + k_coke
            if k_tot * dt > 0.1:
                ratio = 0.1/(k_tot*dt)
                k_dist *= ratio; k_coke *= ratio

            if r0 > 1e-6 and g > 0.01:
                r_total = k_tot * (r0 ** (order - 1.0))
                dR = g * r0 * r_total * dt
                if dR > r0: dR = r0
                aR[k] = r0 - dR

                dC = (rho_vr * g * (r0 ** order) * k_coke * dt) / rho_coke
                aC[k] += dC
                max_coke = 1.0 - porosity_min
                if aC[k] > max_coke: aC[k] = max_coke
                gamma[k] = 1.0 - aC[k]

                aD[k] += (rho_vr * g * (r0 ** order) * k_dist * dt) / rho_dist
                maxD = gamma[k] - aR[k]
                if maxD < 0.0: maxD = 0.0
                if aD[k] > maxD: aD[k] = maxD

        return T, aR, aC, aD, gamma
else:
    # python-only версия с той же логикой
    def _nb_step_advect_react(*args, **kwargs):
        # просто переиспользуем реализацию из блока выше через копию кода без @njit
        T, aR, aC, aD, gamma, \
        dt, rho_vr, rho_coke, rho_dist, \
        T_wall, tau_z, vR, vD, dz, \
        T1_C, T2_C, T_onset_C, \
        A1d, E1d, A1c, E1c, o1, \
        A15d, E15d, A15c, E15c, o15, \
        A2d, E2d, A2c, E2c, o2, \
        scale_dist, scale_coke, porosity_min, phi_por = args

        R = 8.314462618
        NZ = T.shape[0]
        for k in range(NZ):
            T[k] += (T_wall - T[k]) * (dt / tau_z[k])

        sigR = max(0.0, vR) * dt / dz
        sigD = max(0.0, vD) * dt / dz
        nsub = int(max(1.0, math.ceil(max(sigR, sigD))))
        for _ in range(nsub):
            dt_sub = dt / nsub
            sR = max(0.0, vR) * dt_sub / dz
            porous = gamma < 0.95
            avg_por = max(float(np.mean(gamma[porous])) if np.any(porous) else 1.0, porosity_min)
            sD = max(0.0, vD / avg_por) * dt_sub / dz

            aR_in, aD_in = gamma[0], 0.0
            aR0 = aR.copy(); aD0 = aD.copy()
            prevR, prevD = aR_in, aD_in
            for k in range(NZ):
                curR, curD = aR0[k], aD0[k]
                aR[k] = curR - sR * (curR - prevR)
                aD[k] = curD - sD * (curD - prevD)
                if aR[k] < 0.0: aR[k] = 0.0
                if aR[k] > gamma[k]: aR[k] = gamma[k]
                if aD[k] < 0.0: aD[k] = 0.0
                prevR, prevD = curR, curD

        for k in range(NZ):
            Tc = T[k]
            if Tc < T_onset_C: continue
            if Tc < T1_C:   order, Ad, Ed, Ac, Ec = o1,  A1d,  E1d,  A1c,  E1c
            elif Tc < T2_C: order, Ad, Ed, Ac, Ec = o15, A15d, E15d, A15c, E15c
            else:           order, Ad, Ed, Ac, Ec = o2,  A2d,  E2d,  A2c,  E2c

            Tk = Tc + 273.15
            k_dist = scale_dist * Ad * math.exp(-Ed / (R * Tk))
            k_coke = scale_coke * Ac * math.exp(-Ec / (R * Tk))
            g = gamma[k]; r0 = aR[k]

            if g < 0.4:
                factor = phi_por + (1.0 - phi_por) * (g/0.4)
                k_dist *= factor; k_coke *= factor

            k_tot = k_dist + k_coke
            if k_tot*dt > 0.1:
                ratio = 0.1/(k_tot*dt)
                k_dist *= ratio; k_coke *= ratio

            if r0 > 1e-6 and g > 0.01:
                r_total = (k_dist + k_coke) * (r0 ** (order - 1.0))
                dR = g * r0 * r_total * dt
                if dR > r0: dR = r0
                aR[k] = r0 - dR
                dC = (rho_vr * g * (r0 ** order) * k_coke * dt) / rho_coke
                aC[k] += dC
                max_coke = 1.0 - porosity_min
                if aC[k] > max_coke: aC[k] = max_coke
                gamma[k] = 1.0 - aC[k]
                aD[k] += (rho_vr * g * (r0 ** order) * k_dist * dt) / rho_dist
                maxD = gamma[k] - aR[k];  maxD = 0.0 if maxD < 0 else maxD
                if aD[k] > maxD: aD[k] = maxD

        return T, aR, aC, aD, gamma

class Coking1DSolver:
    def __init__(self, geom: Geometry, inlet: Inlet, walls: Walls,
                 mats: Materials, tcfg: TimeSetup, kin: VR3Kinetics,
                 wall_energy: Optional[WallEnergy] = None,
                 mix_energy: Optional[MixtureEnergy] = None):
        self.g, self.inlet, self.walls, self.mats, self.tcfg, self.kin = geom, inlet, walls, mats, tcfg, kin
        self.wall_energy = wall_energy
        self.mix_energy = mix_energy
        self.energy_enabled = (wall_energy is not None) and (mix_energy is not None)

        NZ = geom.NZ
        self.aR = np.ones(NZ, dtype=np.float64)
        self.aC = np.zeros(NZ, dtype=np.float64)
        self.aD = np.zeros(NZ, dtype=np.float64)
        self.gamma = 1.0 - self.aC

        self.porosity_min = float(getattr(mats, 'porosity_min', 0.3))
        self.time_s = 0.0
        self.vol_cell = self.g.A * max(self.g.dz, 1e-9)

        self.k_mix_gas = 0.05
        self.current_h_mix = 0.0

        if self.energy_enabled:
            self._init_energy_state()
        else:
            # температура храним в °C для обратной совместимости
            self.T = np.full(NZ, inlet.T_in_C, dtype=np.float64)
            self.tau_z = _make_tau_profile(self.g, self.walls)
            k_ht = 0.02
            ht_mult = 1.0 + k_ht * float(self.inlet.v_gas_base_factor)
            self.tau_z = self.tau_z / max(ht_mult, 1e-6)

        self.m_vr0 = float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

        # снимки/истории
        self.snap_times_h = list(tcfg.snapshots_h); self._snap_idx = 0
        self.snapshots = {"t_h": [], "T": [], "aR": [], "aD": [], "aC": []}

        self.time_h_hist = []; self.bed_eq_cm_hist = []; self.bed_front_cm_hist = []
        self.contour_every_s = float(tcfg.contour_every_s)
        self.contour_t, self.contour_aR, self.contour_aD, self.contour_aC, self.contour_T = [], [], [], [], []
        self.porosity_avg = 1.0

        # истории по стенке/смеси
        if self.energy_enabled:
            self.shell_hist_t: list[float] = []
            self.shell_hist_out: list[np.ndarray] = []
            self.shell_hist_in: list[np.ndarray] = []
            self.shell_meas_t: list[float] = []
            self.shell_meas_upper: list[float] = []
            self.shell_meas_lower: list[float] = []
        else:
            self.shell_hist_t = []
            self.shell_hist_out = []
            self.shell_hist_in = []
            self.shell_meas_t = []
            self.shell_meas_upper = []
            self.shell_meas_lower = []

        # первый снимок
        self._take_snapshot(); self._maybe_take_contour(force=True)
        if self.snap_times_h and abs(self.snapshots["t_h"][-1]-self.snap_times_h[0]) < 1e-6:
            self._snap_idx = 1

    # сервис
    def bed_height_front(self, thr: float = 0.05) -> float:
        idx = np.where(self.aC > thr)[0]
        if idx.size == 0: return 0.0
        top_idx = idx[-1] + 1
        return float(min(top_idx * self.g.dz, self.g.H))

    def bed_height_equiv(self) -> float:
        eps_max = 1.0 - self.porosity_min
        if eps_max <= 1e-12: return 0.0
        aC_clip = np.clip(self.aC, 0.0, eps_max)
        return float(min(np.sum(aC_clip/eps_max) * self.g.dz, self.g.H))

    def coke_mass(self) -> float:
        return float(np.sum(self.mats.rho_coke_bulk * self.aC) * self.vol_cell)

    def inlet_mass_total(self) -> float:
        return float(self.inlet.m_dot_kg_s * self.time_s)

    def vr_inventory_mass(self) -> float:
        return float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

    def get_porosity_profile(self) -> np.ndarray:
        return np.maximum(self.gamma, self.porosity_min)

    def _init_energy_state(self):
        NZ = self.g.NZ
        Tin_K = float(self.inlet.T_in_C) + 273.15
        self.T_mix_K = np.full(NZ, Tin_K, dtype=np.float64)
        self.reaction_heat_W_m3 = np.zeros(NZ, dtype=np.float64)
        self.perimeter = math.pi * max(self.g.D, 1e-6)

        self.wall_zones = int(max(getattr(self.wall_energy, "zones", 3), 1))
        self._setup_wall_geometry()

        T_amb_K = float(self.wall_energy.T_amb_C) + 273.15
        self.wall_T_amb_K = T_amb_K
        self.wall_T_sky_K = max(T_amb_K - 10.0, 200.0)
        self.T_shell_out_K = np.full(self.wall_zones, T_amb_K, dtype=np.float64)
        self.T_shell_in_K = np.full(self.wall_zones, Tin_K, dtype=np.float64)
        self.shell_obs_out_K = np.full(self.wall_zones, np.nan, dtype=np.float64)

        self.current_h_mix = h_mix(
            self.inlet.m_dot_kg_s,
            getattr(self.inlet, "p_kg_cm2", 1.0),
            self.mix_energy.h0_mix,
            self.mix_energy.alpha_mdot,
            self.mix_energy.alpha_p,
            self.mix_energy.mdot_ref,
            self.mix_energy.p_ref,
        )

        self._update_temperature_cache()

    def _setup_wall_geometry(self):
        radius = max(self.g.D, 1e-9) / 2.0
        head_height = min(radius, self.g.H / 2.0)
        if self.g.H > 0.0:
            head_height = min(head_height, self.g.H / 3.0)
        else:
            head_height = 0.0
        cyl_height = max(self.g.H - 2.0 * head_height, 0.0)

        self.zone_areas = np.zeros(self.wall_zones, dtype=np.float64)
        self.zone_lengths = np.zeros(self.wall_zones, dtype=np.float64)
        self.zone_eps = np.full(self.wall_zones, float(self.wall_energy.outer.epsilon), dtype=np.float64)

        if self.wall_zones >= 3:
            head_area = math.pi * radius ** 2
            cyl_area = 2.0 * math.pi * radius * max(cyl_height, 0.0)
            self.zone_areas[0] = head_area
            self.zone_areas[2] = head_area
            self.zone_areas[1] = max(cyl_area, 1e-9)
            self.zone_lengths[0] = head_height
            self.zone_lengths[2] = head_height
            self.zone_lengths[1] = max(cyl_height, 1e-9)
        else:
            total_area = 2.0 * math.pi * radius * max(self.g.H, 1e-9)
            for z in range(self.wall_zones):
                self.zone_areas[z] = total_area / self.wall_zones
                self.zone_lengths[z] = max(self.g.H / self.wall_zones, 1e-9)

        outer = self.wall_energy.outer
        inner = self.wall_energy.inner
        self.wall_C_out = np.maximum(self.zone_areas * outer.thickness * outer.rho * outer.cp, 1e-6)
        self.wall_C_in = np.maximum(self.zone_areas * inner.thickness * inner.rho * inner.cp, 1e-6)

        self.wall_K = np.zeros(self.wall_zones, dtype=np.float64)
        for z in range(self.wall_zones):
            A = max(self.zone_areas[z], 1e-9)
            R = outer.thickness / max(outer.k * A, 1e-9) + inner.thickness / max(inner.k * A, 1e-9)
            self.wall_K[z] = 1.0 / max(R, 1e-9)

        # карта соответствия ячеек зонам
        self.cell_zone_map = np.zeros(self.g.NZ, dtype=np.int32)
        z_nodes = self.g.z
        for idx, z in enumerate(z_nodes):
            if self.wall_zones < 3:
                self.cell_zone_map[idx] = 0
            else:
                if z <= head_height + 1e-12:
                    self.cell_zone_map[idx] = 0
                elif z >= self.g.H - head_height - 1e-12:
                    self.cell_zone_map[idx] = self.wall_zones - 1
                else:
                    self.cell_zone_map[idx] = 1

        self.zone_cell_indices = []
        for z in range(self.wall_zones):
            indices = np.where(self.cell_zone_map == z)[0]
            if indices.size == 0:
                indices = np.arange(self.g.NZ)
            self.zone_cell_indices.append(indices)

        self.zone_bottom = 0
        self.zone_top = self.wall_zones - 1

    def _update_temperature_cache(self):
        if self.energy_enabled:
            self.T = self.T_mix_K - 273.15
        else:
            self.T = np.asarray(self.T, dtype=np.float64)

    def update_porosity_effects(self):
        zone = self.aC > 0.01
        self.porosity_avg = max(float(np.mean(self.gamma[zone])) if np.any(zone) else 1.0,
                                self.porosity_min)

    def coke_yield_pct_feed(self) -> float:
        return 100.0 * self.coke_mass() / max(self.inlet_mass_total(), 1e-9)

    def coke_yield_pct_balance(self) -> float:
        return 100.0 * self.coke_mass() / max(self.m_vr0 + self.inlet_mass_total(), 1e-9)

    def _take_snapshot(self):
        t_h = self.time_s/3600.0
        self.snapshots["t_h"].append(t_h)
        self.snapshots["T"].append(self.T.copy())
        self.snapshots["aR"].append(self.aR.copy())
        self.snapshots["aD"].append(self.aD.copy())
        self.snapshots["aC"].append(self.aC.copy())

    def _maybe_take_snapshot(self):
        if self._snap_idx >= len(self.snap_times_h): return
        t_h = self.time_s/3600.0
        if t_h + 1e-4 >= self.snap_times_h[self._snap_idx]:
            self._take_snapshot(); self._snap_idx += 1

    def _maybe_take_contour(self, force: bool = False):
        if force or (len(self.contour_t) == 0) or (self.time_s - self.contour_t[-1] >= self.contour_every_s - 1e-12):
            self.contour_t.append(self.time_s)
            self.contour_T.append(self.T.copy())
            self.contour_aR.append(self.aR.copy())
            self.contour_aD.append(self.aD.copy())
            self.contour_aC.append(self.aC.copy())

    def _advect(self, vR_eff: float, vD_gas: float):
        dt = float(self.tcfg.dt)
        dz = max(self.g.dz, 1e-9)
        sigR = max(vR_eff, 0.0) * dt / dz
        sigD = max(vD_gas, 0.0) * dt / dz
        nsub = max(1, int(math.ceil(max(sigR, sigD, 1.0))))

        for _ in range(nsub):
            dt_sub = dt / nsub
            sR = max(vR_eff, 0.0) * dt_sub / dz
            avg_por = max(self.porosity_avg, self.porosity_min)
            sD = max(vD_gas, 0.0) * dt_sub / (dz * max(avg_por, 1e-3))

            aR_prev = float(self.gamma[0])
            aD_prev = 0.0
            aR0 = self.aR.copy()
            aD0 = self.aD.copy()
            for k in range(self.g.NZ):
                curR = aR0[k]
                curD = aD0[k]
                self.aR[k] = curR - sR * (curR - aR_prev)
                self.aD[k] = curD - sD * (curD - aD_prev)
                if self.aR[k] < 0.0:
                    self.aR[k] = 0.0
                if self.aR[k] > self.gamma[k]:
                    self.aR[k] = self.gamma[k]
                if self.aD[k] < 0.0:
                    self.aD[k] = 0.0
                aR_prev = curR
                aD_prev = curD

            for k in range(self.g.NZ):
                maxD = self.gamma[k] - self.aR[k]
                if maxD < 0.0:
                    maxD = 0.0
                if self.aD[k] > maxD:
                    self.aD[k] = maxD

    def _reaction_step(self, dt: float):
        rho_vr = float(self.inlet.rho_vr)
        rho_coke = max(float(self.mats.rho_coke_bulk), 1e-9)
        rho_dist = max(float(self.mats.rho_dist_vap), 1e-9)
        self.reaction_heat_W_m3.fill(0.0)

        T1 = self.kin.T1_C + self.kin.dT1
        T2 = self.kin.T2_C + self.kin.dT2

        for k in range(self.g.NZ):
            gamma_k = max(self.gamma[k], self.porosity_min)
            r0 = self.aR[k]
            if r0 <= 1e-9 or gamma_k <= 1e-6:
                continue

            Tc = float(self.T_mix_K[k] - 273.15)
            if Tc < T_COKE_ONSET_C:
                continue

            if Tc < T1:
                regime = self.kin.reg1
            elif Tc < T2:
                regime = self.kin.reg15
            else:
                regime = self.kin.reg2

            Tk = Tc + 273.15
            k_dist = self.kin.scale_dist * self.kin.A_dist_scale * regime.A_dist * math.exp(-regime.Ea_dist / (8.314462618 * Tk))
            k_coke = self.kin.scale_coke * self.kin.A_coke_scale * regime.A_coke * math.exp(-regime.Ea_coke / (8.314462618 * Tk))

            phi = self.kin.phi_por
            if gamma_k < 0.4:
                factor = phi + (1.0 - phi) * (gamma_k / 0.4)
                k_dist *= factor
                k_coke *= factor

            k_tot = k_dist + k_coke
            if k_tot <= 0.0:
                continue

            if k_tot * dt > 0.1:
                scale = 0.1 / (k_tot * dt)
                k_dist *= scale
                k_coke *= scale
                k_tot = k_dist + k_coke

            order = regime.order
            base = r0 ** max(order - 1.0, 0.0)
            r_total = k_tot * base
            dR = gamma_k * r0 * r_total * dt
            if dR > r0:
                dR = r0
            if dR <= 0.0:
                continue

            self.aR[k] = max(r0 - dR, 0.0)

            mass_factor = rho_vr * gamma_k * (r0 ** order) * dt
            m_dist = mass_factor * k_dist
            m_coke = mass_factor * k_coke

            dC = m_coke / rho_coke
            self.aC[k] += dC
            max_coke = 1.0 - self.porosity_min
            if self.aC[k] > max_coke:
                self.aC[k] = max_coke

            self.gamma[k] = max(1.0 - self.aC[k], self.porosity_min)
            if self.aR[k] > self.gamma[k]:
                self.aR[k] = self.gamma[k]

            dD = m_dist / rho_dist
            self.aD[k] += dD
            max_dist = self.gamma[k] - self.aR[k]
            if max_dist < 0.0:
                max_dist = 0.0
            if self.aD[k] > max_dist:
                self.aD[k] = max_dist

            if self.mix_energy is not None and dt > 0.0:
                q = -(self.mix_energy.dH_dist * m_dist + self.mix_energy.dH_coke * m_coke) / (self.vol_cell * dt)
                self.reaction_heat_W_m3[k] = q

    def _update_wall_energy(self, h_local: float, dt: float):
        for z in range(self.wall_zones):
            A_zone = max(self.zone_areas[z], 1e-9)
            C_out = self.wall_C_out[z]
            C_in = self.wall_C_in[z]
            K_wall = self.wall_K[z]

            obs = self.shell_obs_out_K[z]
            T_out = self.T_shell_out_K[z]
            if np.isfinite(obs):
                T_out = obs

            T_in = self.T_shell_in_K[z]
            mix_idx = self.zone_cell_indices[z]
            T_mix_zone = float(np.mean(self.T_mix_K[mix_idx]))

            dT_out_dt = (
                -K_wall * (T_out - T_in)
                - self.wall_energy.h_amb * A_zone * (T_out - self.wall_T_amb_K)
                - SIGMA_SB * self.zone_eps[z] * A_zone * (T_out ** 4 - self.wall_T_sky_K ** 4)
            ) / C_out

            dT_in_dt = (
                K_wall * (T_out - T_in)
                - h_local * A_zone * (T_in - T_mix_zone)
            ) / C_in

            T_out_new = T_out + dt * dT_out_dt
            if np.isfinite(obs):
                T_out_new = 0.5 * (T_out_new + obs)
            self.T_shell_out_K[z] = max(T_out_new, 200.0)
            self.T_shell_in_K[z] = max(T_in + dt * dT_in_dt, 200.0)

    def _update_mixture_energy(self, vR_eff: float, h_local: float, dt: float):
        rho = max(float(self.inlet.rho_vr), 1e-9)
        cp_eff = max(float(self.mix_energy.cp_eff), 1e-9)
        cap = rho * cp_eff
        lam = max(float(self.mix_energy.lambda_eff), 0.0)
        dz = max(self.g.dz, 1e-9)
        T_new = self.T_mix_K.copy()
        Tin_K = float(self.inlet.T_in_C) + 273.15
        Tout_K = float(self.T_mix_K[-1])
        V_cell = max(self.vol_cell, 1e-9)

        for i in range(self.g.NZ):
            Ti = self.T_mix_K[i]

            if self.g.NZ == 1:
                d2 = 0.0
            elif i == 0:
                d2 = 2.0 * (self.T_mix_K[1] - Ti) / (dz ** 2)
            elif i == self.g.NZ - 1:
                d2 = 2.0 * (self.T_mix_K[i - 1] - Ti) / (dz ** 2)
            else:
                d2 = (self.T_mix_K[i + 1] - 2.0 * Ti + self.T_mix_K[i - 1]) / (dz ** 2)

            if vR_eff >= 0.0:
                upstream = Tin_K if i == 0 else self.T_mix_K[i - 1]
                d1 = (Ti - upstream) / dz
            else:
                downstream = Tout_K if i == self.g.NZ - 1 else self.T_mix_K[i + 1]
                d1 = (downstream - Ti) / dz

            zone = self.cell_zone_map[i]
            T_shell_in = self.T_shell_in_K[zone]
            heat_transfer = h_local * self.perimeter / V_cell * (T_shell_in - Ti)
            q_react = self.reaction_heat_W_m3[i]

            numerator = lam * d2 - rho * cp_eff * vR_eff * d1 + heat_transfer + q_react
            T_new[i] = Ti + dt * numerator / cap

        self.T_mix_K = np.clip(T_new, 200.0, 2000.0)
        self._update_temperature_cache()

    def set_shell_observations(self, t: float, T_upper_C: Optional[float], T_lower_C: Optional[float]):
        """Обновляет целевые внешние температуры оболочки."""
        self.shell_meas_t.append(float(t))
        self.shell_meas_upper.append(float(T_upper_C) if T_upper_C is not None else float('nan'))
        self.shell_meas_lower.append(float(T_lower_C) if T_lower_C is not None else float('nan'))
        if not self.energy_enabled:
            return

        if T_upper_C is None:
            self.shell_obs_out_K[self.zone_top] = float('nan')
        else:
            self.shell_obs_out_K[self.zone_top] = float(T_upper_C) + 273.15

        if T_lower_C is None:
            self.shell_obs_out_K[self.zone_bottom] = float('nan')
        else:
            self.shell_obs_out_K[self.zone_bottom] = float(T_lower_C) + 273.15

    def step(self):
        self.update_porosity_effects()

        vR_base = self.inlet.velocity(self.g)
        vD_gas = self.inlet.velocity_gas(self.g, self.porosity_avg)
        vR_eff = vR_base + self.k_mix_gas * vD_gas

        if self.energy_enabled:
            dt = float(self.tcfg.dt)
            self._advect(vR_eff, vD_gas)
            self._reaction_step(dt)
            self.current_h_mix = h_mix(
                self.inlet.m_dot_kg_s,
                getattr(self.inlet, "p_kg_cm2", 1.0),
                self.mix_energy.h0_mix,
                self.mix_energy.alpha_mdot,
                self.mix_energy.alpha_p,
                self.mix_energy.mdot_ref,
                self.mix_energy.p_ref,
            )
            self._update_wall_energy(self.current_h_mix, dt)
            self._update_mixture_energy(vR_eff, self.current_h_mix, dt)
            self.update_porosity_effects()
            self.time_s += dt

            self.shell_hist_t.append(self.time_s)
            self.shell_hist_out.append(self.T_shell_out_K.copy())
            self.shell_hist_in.append(self.T_shell_in_K.copy())
        else:
            r1, r15, r2 = self.kin.reg1, self.kin.reg15, self.kin.reg2
            self.T, self.aR, self.aC, self.aD, self.gamma = _nb_step_advect_react(
                self.T, self.aR, self.aC, self.aD, self.gamma,
                self.tcfg.dt,
                self.inlet.rho_vr, self.mats.rho_coke_bulk, self.mats.rho_dist_vap,
                self.walls.T_wall_C, self.tau_z,
                vR_eff,
                vD_gas,
                self.g.dz,
                self.kin.T1_C + self.kin.dT1,
                self.kin.T2_C + self.kin.dT2,
                T_COKE_ONSET_C,
                r1.A_dist * self.kin.A_dist_scale, r1.Ea_dist, r1.A_coke * self.kin.A_coke_scale, r1.Ea_coke, r1.order,
                r15.A_dist * self.kin.A_dist_scale, r15.Ea_dist, r15.A_coke * self.kin.A_coke_scale, r15.Ea_coke, r15.order,
                r2.A_dist * self.kin.A_dist_scale, r2.Ea_dist, r2.A_coke * self.kin.A_coke_scale, r2.Ea_coke, r2.order,
                self.kin.scale_dist, self.kin.scale_coke,
                self.porosity_min,
                self.kin.phi_por,
            )
            self.time_s += self.tcfg.dt

    def run(self, verbose_hourly: bool = True):
        steps = int(self.tcfg.total_hours * 3600.0 / self.tcfg.dt)
        next_hour_s = 3600.0
        for _ in range(steps):
            self.step()
            if self.time_s + 1e-12 >= next_hour_s:
                self.time_h_hist.append(self.time_s/3600.0)
                self.bed_eq_cm_hist.append(self.bed_height_equiv()*100.0)
                self.bed_front_cm_hist.append(self.bed_height_front()*100.0)
                if verbose_hourly:
                    y_feed = self.coke_yield_pct_feed(); y_bal = self.coke_yield_pct_balance()
                    if self.time_s < 2*3600.0:
                        print(f"t = {self.time_s/3600.0:5.1f} ч | H_eq={self.bed_eq_cm_hist[-1]:.1f} см | "
                              f"H_front={self.bed_front_cm_hist[-1]:.1f} см | Yбал={y_bal:5.2f}% | "
                              f"T_avg={np.mean(self.T):.1f}°C | ε_avg={self.porosity_avg:.3f}")
                    else:
                        print(f"t = {self.time_s/3600.0:5.1f} ч | H_eq={self.bed_eq_cm_hist[-1]:.1f} см | "
                              f"H_front={self.bed_front_cm_hist[-1]:.1f} см | Yfeed={y_feed:5.2f}% | "
                              f"Yбал={y_bal:5.2f}% | T_avg={np.mean(self.T):.1f}°C | ε_avg={self.porosity_avg:.3f}")
                next_hour_s += 3600.0
            self._maybe_take_snapshot(); self._maybe_take_contour()

        result = {
            "H_bed_m": self.bed_height_equiv(),
            "H_front_m": self.bed_height_front(),
            "yield_pct": self.coke_yield_pct_feed(),
            "T_avg_C": float(np.mean(self.T)),
            "porosity_avg": float(self.porosity_avg),
            "final": {"T": self.T.copy(), "aR": self.aR.copy(), "aD": self.aD.copy(), "aC": self.aC.copy()},
            "z": self.g.z.copy(),
            "snapshots": {
                "t_h": np.array(self.snapshots["t_h"], dtype=float),
                "T":   np.stack(self.snapshots["T"], axis=0),
                "aR":  np.stack(self.snapshots["aR"], axis=0),
                "aD":  np.stack(self.snapshots["aD"], axis=0),
                "aC":  np.stack(self.snapshots["aC"], axis=0),
            },
            "growth": {
                "t_h": np.array(self.time_h_hist, dtype=float),
                "H_cm": np.array(self.bed_eq_cm_hist, dtype=float),
                "H_front_cm": np.array(self.bed_front_cm_hist, dtype=float),
            },
            "contours": {
                "t_s": np.array(self.contour_t, dtype=float),
                "T":   np.stack(self.contour_T, axis=0),
                "aR":  np.stack(self.contour_aR, axis=0),
                "aD":  np.stack(self.contour_aD, axis=0),
                "aC":  np.stack(self.contour_aC, axis=0),
            },
        }
        if self.energy_enabled:
            result["shell"] = {
                "t_s": np.array(self.shell_hist_t, dtype=float),
                "T_out_C": np.array([arr - 273.15 for arr in self.shell_hist_out], dtype=float),
                "T_in_C": np.array([arr - 273.15 for arr in self.shell_hist_in], dtype=float),
                "measurements": {
                    "t": np.array(self.shell_meas_t, dtype=float),
                    "upper_C": np.array(self.shell_meas_upper, dtype=float),
                    "lower_C": np.array(self.shell_meas_lower, dtype=float),
                },
            }
        return result
