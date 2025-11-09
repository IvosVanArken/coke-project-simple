#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Калибровка 1D-модели замедленного коксования по промышленным временным рядам."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - требуется для работы
    raise SystemExit("Необходим пакет numpy (отсутствует в окружении)") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - среда без matplotlib
    plt = None

from src.geometry import Geometry
from src.kinetics import VR3Kinetics
from src.params import (
    Inlet,
    Materials,
    MixtureEnergy,
    TimeSetup,
    WallEnergy,
    WallLayer,
    Walls,
)
from src.solver_1d import Coking1DSolver

EXCEL_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
COLUMN_ALIASES = {
    "timestamp": "timestamp",
    "Время": "timestamp",
    "Давление": "pressure",
    "Температура входа": "T_in",
    "Температура выхода": "T_out",
    "Температура верхнего днища": "T_head_upper",
    "Температура нижнего днища": "T_head_lower",
    "Расход": "flow",
}
PARAM_SPECS: tuple[tuple[str, float, float], ...] = (
    ("outer_k", 1.0, 40.0),
    ("outer_rho", 3000.0, 8000.0),
    ("outer_cp", 400.0, 900.0),
    ("outer_thickness", 0.01, 0.25),
    ("outer_epsilon", 0.6, 0.95),
    ("inner_k", 1.0, 40.0),
    ("inner_rho", 3000.0, 8000.0),
    ("inner_cp", 400.0, 900.0),
    ("inner_thickness", 0.01, 0.25),
    ("h_amb", 5.0, 50.0),
    ("T_amb_C", 0.0, 50.0),
    ("lambda_eff", 0.1, 3.0),
    ("cp_eff", 1500.0, 3500.0),
    ("h0_mix", 20.0, 800.0),
    ("alpha_mdot", 0.2, 0.9),
    ("alpha_p", 0.0, 0.4),
    ("mdot_ref", 0.1, 20.0),
    ("p_ref", 0.5, 20.0),
    ("dH_dist", -3.0e5, 3.0e5),
    ("dH_coke", -3.0e5, 3.0e5),
    ("A_dist_scale", 0.3, 3.0),
    ("A_coke_scale", 0.3, 3.0),
    ("dT1", -20.0, 20.0),
    ("dT2", -20.0, 20.0),
    ("phi_por", 0.3, 1.0),
)
W_OUT, W_TOP, W_BOT, W_H = 0.35, 0.30, 0.30, 0.05
BIG_PENALTY = 1.0e6


def excel_serial_to_datetime(value: float) -> datetime:
    origin = datetime(1899, 12, 30)
    return origin + timedelta(days=float(value))


def _col_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - 64)
    return idx - 1


def read_excel_basic(path: Path) -> tuple[list[str], list[list[float]]]:
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("s:si", EXCEL_NS):
                fragments = [node.text for node in si.findall(".//s:t", EXCEL_NS) if node.text]
                shared.append("".join(fragments))
        sheet_xml = zf.read("xl/worksheets/sheet1.xml")

    root = ET.fromstring(sheet_xml)
    sheet_data = root.find("s:sheetData", EXCEL_NS)
    if sheet_data is None:
        raise RuntimeError("В Excel не найден sheetData")

    headers: list[str] = []
    data: list[list[float]] = []
    max_cols = 0
    for row_idx, row in enumerate(sheet_data.findall("s:row", EXCEL_NS)):
        values: dict[int, float] = {}
        for cell in row.findall("s:c", EXCEL_NS):
            ref = cell.get("r")
            if ref is None:
                continue
            idx = _col_to_index(ref)
            node = cell.find("s:v", EXCEL_NS)
            if node is None or node.text is None:
                continue
            if cell.get("t") == "s":
                try:
                    s_idx = int(node.text)
                    values[idx] = shared[s_idx]
                except Exception:
                    values[idx] = ""
            else:
                try:
                    values[idx] = float(node.text)
                except ValueError:
                    values[idx] = math.nan
        if row_idx == 0:
            max_cols = max(values.keys(), default=-1) + 1
            headers = ["" for _ in range(max_cols)]
            for col, val in values.items():
                headers[col] = str(val)
            continue
        row_list: list[float] = [math.nan] * max_cols
        for col, val in values.items():
            if col < max_cols:
                row_list[col] = float(val) if isinstance(val, (int, float)) else math.nan
        data.append(row_list)
    return headers, data


def interpolate_series(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    mask = np.isfinite(times) & np.isfinite(values)
    if np.count_nonzero(mask) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    t_valid = times[mask]
    v_valid = values[mask]
    order = np.argsort(t_valid)
    return np.interp(grid, t_valid[order], v_valid[order], left=v_valid[order][0], right=v_valid[order][-1])


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-6)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    err = np.abs((y_pred[mask] - y_true[mask]) / np.maximum(np.abs(y_true[mask]), 1e-6))
    return float(np.mean(err) * 100.0)


def differential_evolution(cost_fn, bounds: Sequence[tuple[float, float]], *, initial: Optional[np.ndarray] = None,
                            max_generations: int = 60, pop_size: int = 18, mutation: tuple[float, float] = (0.6, 0.9),
                            crossover: float = 0.7, seed: int = 42) -> tuple[np.ndarray, float]:
    rng = random.Random(seed)
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    pop = np.zeros((pop_size, dim), dtype=float)
    scores = np.full(pop_size, float("inf"))

    for i in range(pop_size):
        if initial is not None and i == 0:
            pop[i] = np.clip(initial, lower, upper)
        else:
            for j in range(dim):
                pop[i, j] = rng.uniform(lower[j], upper[j])
        scores[i] = cost_fn(pop[i])
    best_idx = int(np.argmin(scores))
    best_vec = pop[best_idx].copy()
    best_score = float(scores[best_idx])

    for gen in range(max_generations):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = rng.sample(idxs, 3)
            F = rng.uniform(*mutation)
            mutant = pop[a] + F * (pop[b] - pop[c])
            trial = pop[i].copy()
            j_rand = rng.randrange(dim)
            for j in range(dim):
                if rng.random() < crossover or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lower, upper)
            score = cost_fn(trial)
            if score < scores[i]:
                pop[i] = trial
                scores[i] = score
                if score < best_score:
                    best_score = float(score)
                    best_vec = trial.copy()
        print(f"[DE] поколение {gen + 1}/{max_generations}: лучшая цель = {best_score:.4f}")
    return best_vec, best_score


def _nan_to_none(val: float) -> Optional[float]:
    return float(val) if math.isfinite(val) else None


class CalibrationContext:
    def __init__(self, excel_path: Path, t_start: Optional[datetime], t_end: Optional[datetime]):
        headers, rows = read_excel_basic(excel_path)
        alias_idx: dict[str, int] = {}
        for idx, name in enumerate(headers):
            alias = COLUMN_ALIASES.get(name.strip())
            if alias is not None:
                alias_idx[alias] = idx

        timestamps: list[datetime] = []
        series = {key: [] for key in COLUMN_ALIASES.values() if key != "timestamp"}
        ts_col = alias_idx.get("timestamp")
        if ts_col is None:
            raise RuntimeError("В Excel не найден столбец времени")

        for row in rows:
            ts_val = row[ts_col] if ts_col < len(row) else math.nan
            if not math.isfinite(ts_val):
                continue
            timestamps.append(excel_serial_to_datetime(ts_val))
            for alias, idx in alias_idx.items():
                if alias == "timestamp":
                    continue
                value = row[idx] if idx < len(row) else math.nan
                series.setdefault(alias, []).append(float(value) if math.isfinite(value) else math.nan)

        if not timestamps:
            raise RuntimeError("Пустой временной ряд в Excel")
        if t_start is None:
            t_start = timestamps[0]
        if t_end is None:
            t_end = timestamps[-1]
        if t_end <= t_start:
            raise ValueError("t_end должно быть больше t_start")

        use_indices = [i for i, ts in enumerate(timestamps) if t_start <= ts <= t_end]
        if not use_indices:
            raise RuntimeError("Нет точек в указанном интервале времени")

        ts_sel = [timestamps[i] for i in use_indices]
        t0 = ts_sel[0]
        self.times_sec = np.array([(ts - t0).total_seconds() for ts in ts_sel], dtype=float)
        self.duration_s = float((ts_sel[-1] - t0).total_seconds())

        def pick(name: str) -> np.ndarray:
            data = series.get(name, [])
            return np.array([data[i] if i < len(data) else math.nan for i in use_indices], dtype=float)

        self.T_in_meas = pick("T_in")
        self.T_out_meas = pick("T_out")
        self.T_upper_meas = pick("T_head_upper")
        self.T_lower_meas = pick("T_head_lower")
        self.pressure_meas = pick("pressure")
        self.flow_meas = pick("flow")

        self.dt = 30.0
        steps = int(math.ceil(self.duration_s / self.dt))
        self.grid_times = np.linspace(0.0, steps * self.dt, steps + 1)

        self.T_in_grid = interpolate_series(self.times_sec, self.T_in_meas, self.grid_times)
        self.flow_grid = interpolate_series(self.times_sec, self.flow_meas, self.grid_times)
        self.pressure_grid = interpolate_series(self.times_sec, self.pressure_meas, self.grid_times)
        self.upper_grid = interpolate_series(self.times_sec, self.T_upper_meas, self.grid_times)
        self.lower_grid = interpolate_series(self.times_sec, self.T_lower_meas, self.grid_times)

        self.rho_feed = 1050.0
        self.grid_mdot = self.flow_grid * self.rho_feed / 3600.0
        self.avg_mdot = float(np.nanmean(self.grid_mdot)) if np.isfinite(np.nanmean(self.grid_mdot)) else 5.0
        self.avg_pressure = float(np.nanmean(self.pressure_grid)) if np.isfinite(np.nanmean(self.pressure_grid)) else 5.0

        self.geom = Geometry(H=21.25, D=5.5, NZ=Geometry().NZ)
        self.walls = Walls()
        self.materials = Materials()
        total_hours = self.grid_times[-1] / 3600.0
        self.time_setup = TimeSetup(total_hours=total_hours, dt=self.dt, snapshots_h=(), contour_every_s=self.duration_s + self.dt)
        self.base_inlet = Inlet(
            T_in_C=float(self.T_in_grid[0]) if math.isfinite(self.T_in_grid[0]) else 360.0,
            m_dot_kg_s=float(self.grid_mdot[0]) if math.isfinite(self.grid_mdot[0]) else self.avg_mdot,
            rho_vr=self.rho_feed,
            v_gas_base_factor=8.0,
            p_kg_cm2=float(self.pressure_grid[0]) if math.isfinite(self.pressure_grid[0]) else self.avg_pressure,
        )

        self.initial_guess = {
            "outer_k": 12.0,
            "outer_rho": 5200.0,
            "outer_cp": 650.0,
            "outer_thickness": 0.08,
            "outer_epsilon": 0.85,
            "inner_k": 3.0,
            "inner_rho": 4200.0,
            "inner_cp": 700.0,
            "inner_thickness": 0.12,
            "h_amb": 18.0,
            "T_amb_C": 25.0,
            "lambda_eff": 0.8,
            "cp_eff": 2200.0,
            "h0_mix": 220.0,
            "alpha_mdot": 0.55,
            "alpha_p": 0.15,
            "mdot_ref": max(self.avg_mdot, 0.5),
            "p_ref": max(self.avg_pressure, 1.0),
            "dH_dist": 0.0,
            "dH_coke": 0.0,
            "A_dist_scale": 1.0,
            "A_coke_scale": 1.0,
            "dT1": 0.0,
            "dT2": 0.0,
            "phi_por": 0.6,
        }

    def vector_to_config(self, vec: Sequence[float]):
        if len(vec) != len(PARAM_SPECS):
            return None
        params = {}
        for (name, lower, upper), value in zip(PARAM_SPECS, vec):
            if not math.isfinite(value):
                return None
            params[name] = float(np.clip(value, lower, upper))
        outer = WallLayer(
            k=params["outer_k"],
            rho=params["outer_rho"],
            cp=params["outer_cp"],
            thickness=params["outer_thickness"],
            epsilon=params["outer_epsilon"],
        )
        inner = WallLayer(
            k=params["inner_k"],
            rho=params["inner_rho"],
            cp=params["inner_cp"],
            thickness=params["inner_thickness"],
        )
        wall_energy = WallEnergy(outer=outer, inner=inner, h_amb=params["h_amb"], T_amb_C=params["T_amb_C"], zones=3)
        mix_energy = MixtureEnergy(
            lambda_eff=params["lambda_eff"],
            cp_eff=params["cp_eff"],
            h0_mix=params["h0_mix"],
            alpha_mdot=params["alpha_mdot"],
            alpha_p=params["alpha_p"],
            mdot_ref=max(params["mdot_ref"], 1e-3),
            p_ref=max(params["p_ref"], 1e-3),
            dH_dist=params["dH_dist"],
            dH_coke=params["dH_coke"],
        )
        kin = VR3Kinetics(
            A_dist_scale=params["A_dist_scale"],
            A_coke_scale=params["A_coke_scale"],
            dT1=params["dT1"],
            dT2=params["dT2"],
            phi_por=params["phi_por"],
        )
        return params, wall_energy, mix_energy, kin

    def _prepare_solver(self, wall_energy: WallEnergy, mix_energy: MixtureEnergy, kin: VR3Kinetics) -> Coking1DSolver:
        inlet = replace(self.base_inlet)
        tcfg = replace(self.time_setup)
        return Coking1DSolver(self.geom, inlet, self.walls, self.materials, tcfg, kin,
                               wall_energy=wall_energy, mix_energy=mix_energy)

    def evaluate(self, vec: Sequence[float], record_series: bool = False):
        config = self.vector_to_config(vec)
        if config is None:
            return BIG_PENALTY, None
        params, wall_energy, mix_energy, kin = config
        solver = self._prepare_solver(wall_energy, mix_energy, kin)
        if not solver.energy_enabled:
            return BIG_PENALTY, None

        times = [0.0]
        T_out = [float(solver.T[-1])]
        upper = [float(solver.T_shell_out_K[solver.zone_top] - 273.15)]
        lower = [float(solver.T_shell_out_K[solver.zone_bottom] - 273.15)]
        heights = [float(solver.bed_height_equiv())]

        for i in range(1, len(self.grid_times)):
            idx = i - 1
            Tin = self.T_in_grid[idx]
            mdot = self.grid_mdot[idx]
            pressure = self.pressure_grid[idx]
            upper_meas = self.upper_grid[idx]
            lower_meas = self.lower_grid[idx]

            if math.isfinite(Tin):
                solver.inlet.T_in_C = float(Tin)
            if math.isfinite(mdot):
                solver.inlet.m_dot_kg_s = float(mdot)
            else:
                solver.inlet.m_dot_kg_s = self.avg_mdot
            if math.isfinite(pressure):
                solver.inlet.p_kg_cm2 = float(pressure)
            else:
                solver.inlet.p_kg_cm2 = self.avg_pressure

            solver.set_shell_observations(self.grid_times[idx],
                                          _nan_to_none(upper_meas),
                                          _nan_to_none(lower_meas))
            solver.step()

            times.append(self.grid_times[i])
            T_out.append(float(solver.T[-1]))
            upper.append(float(solver.T_shell_out_K[solver.zone_top] - 273.15))
            lower.append(float(solver.T_shell_out_K[solver.zone_bottom] - 273.15))
            heights.append(float(solver.bed_height_equiv()))

            if not all(math.isfinite(x) for x in (T_out[-1], upper[-1], lower[-1], heights[-1])):
                return BIG_PENALTY, None

        times_arr = np.array(times, dtype=float)
        T_out_arr = np.array(T_out, dtype=float)
        upper_arr = np.array(upper, dtype=float)
        lower_arr = np.array(lower, dtype=float)
        heights_arr = np.array(heights, dtype=float)

        model_out_meas = interpolate_series(times_arr, T_out_arr, self.times_sec)
        model_upper_meas = interpolate_series(times_arr, upper_arr, self.times_sec)
        model_lower_meas = interpolate_series(times_arr, lower_arr, self.times_sec)

        m_out = mape(self.T_out_meas, model_out_meas)
        m_upper = mape(self.T_upper_meas, model_upper_meas)
        m_lower = mape(self.T_lower_meas, model_lower_meas)
        if not all(math.isfinite(x) for x in (m_out, m_upper, m_lower)):
            return BIG_PENALTY, None

        height_final = float(heights_arr[-1])
        height_penalty = abs(height_final - 17.5) / 17.5
        loss = W_OUT * m_out + W_TOP * m_upper + W_BOT * m_lower + W_H * height_penalty

        metrics = {
            "loss": float(loss),
            "mape_out": float(m_out),
            "mape_upper": float(m_upper),
            "mape_lower": float(m_lower),
            "height_final_m": height_final,
            "height_penalty": float(height_penalty),
        }

        if record_series:
            meas_out_grid = interpolate_series(self.times_sec, self.T_out_meas, times_arr)
            meas_upper_grid = interpolate_series(self.times_sec, self.T_upper_meas, times_arr)
            meas_lower_grid = interpolate_series(self.times_sec, self.T_lower_meas, times_arr)
            metrics["series"] = {
                "t_s": times_arr,
                "T_out_model_C": T_out_arr,
                "T_upper_model_C": upper_arr,
                "T_lower_model_C": lower_arr,
                "H_m": heights_arr,
            }
            metrics["measurements"] = {
                "t_s": self.times_sec,
                "T_out_meas_C": self.T_out_meas,
                "T_upper_meas_C": self.T_upper_meas,
                "T_lower_meas_C": self.T_lower_meas,
            }
            metrics["timeseries"] = {
                "t_s": times_arr,
                "T_out_model_C": T_out_arr,
                "T_out_meas_C": meas_out_grid,
                "T_upper_model_C": upper_arr,
                "T_upper_meas_C": meas_upper_grid,
                "T_lower_model_C": lower_arr,
                "T_lower_meas_C": meas_lower_grid,
                "H_m": heights_arr,
            }
            metrics["params"] = params
        return loss, metrics


def save_json(path: Path, data) -> None:
    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)


def save_timeseries_csv(path: Path, metrics: dict) -> None:
    series = metrics["series"]
    meas = metrics["measurements"]
    def _interp(times: np.ndarray, values: np.ndarray, t_val: float) -> float:
        if len(times) == 0:
            return math.nan
        return float(interpolate_series(times, values, np.array([t_val], dtype=float))[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "T_out_meas_C", "T_out_model_C", "T_upper_meas_C", "T_upper_model_C",
                         "T_lower_meas_C", "T_lower_model_C", "H_m"])
        for t, t_out, t_up, t_low, h in zip(series["t_s"], series["T_out_model_C"],
                                            series["T_upper_model_C"], series["T_lower_model_C"],
                                            series["H_m"]):
            meas_out = _interp(meas["t_s"], meas["T_out_meas_C"], t)
            meas_up = _interp(meas["t_s"], meas["T_upper_meas_C"], t)
            meas_low = _interp(meas["t_s"], meas["T_lower_meas_C"], t)
            writer.writerow([float(t), meas_out, float(t_out), meas_up, float(t_up), meas_low, float(t_low), float(h)])


def render_plots(out_dir: Path, metrics: dict) -> None:
    if plt is None:
        print("[WARN] matplotlib недоступен, графики не будут сохранены")
        return
    series = metrics["series"]
    meas = metrics["measurements"]
    t_model_h = series["t_s"] / 3600.0
    t_meas_h = meas["t_s"] / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    axes[0].plot(t_model_h, series["T_out_model_C"], lw=2.0, label="Модель")
    if len(t_meas_h):
        axes[0].scatter(t_meas_h, meas["T_out_meas_C"], c="black", s=20, label="Измерения")
    axes[0].set_ylabel("T вых., °C")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_model_h, series["T_upper_model_C"], lw=2.0, label="Модель")
    if len(t_meas_h):
        axes[1].scatter(t_meas_h, meas["T_upper_meas_C"], c="black", s=20)
    axes[1].set_ylabel("T верх., °C")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_model_h, series["T_lower_model_C"], lw=2.0, label="Модель")
    if len(t_meas_h):
        axes[2].scatter(t_meas_h, meas["T_lower_meas_C"], c="black", s=20)
    axes[2].set_ylabel("T низ., °C")
    axes[2].set_xlabel("Время, ч")
    axes[2].grid(True, alpha=0.3)
    fig.savefig(out_dir / "temperature_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(t_model_h, series["H_m"], lw=2.0)
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Высота слоя, м")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "height.png", dpi=200)
    plt.close(fig)


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Калибровка модели по временным рядам")
    parser.add_argument("--excel", required=True, type=Path, help="Путь к Excel-файлу с данными")
    parser.add_argument("--t-start", type=parse_datetime, help="Начало интервала (ISO)")
    parser.add_argument("--t-end", type=parse_datetime, help="Конец интервала (ISO)")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="Каталог для отчётов")
    parser.add_argument("--max-iters", type=int, default=40, help="Поколений дифф. эволюции")
    parser.add_argument("--pop-size", type=int, default=18, help="Размер популяции")
    parser.add_argument("--seed", type=int, default=42, help="Зерно генератора")
    args = parser.parse_args(argv)

    context = CalibrationContext(args.excel, args.t_start, args.t_end)
    bounds = [(lo, hi) for _, lo, hi in PARAM_SPECS]
    initial_vec = np.array([np.clip(context.initial_guess[name], lo, hi) for name, lo, hi in PARAM_SPECS], dtype=float)

    def cost_fn(vec: Sequence[float]) -> float:
        loss, _ = context.evaluate(vec, record_series=False)
        return float(loss)

    best_vec, _ = differential_evolution(cost_fn, bounds, initial=initial_vec,
                                         max_generations=args.max_iters, pop_size=args.pop_size,
                                         seed=args.seed)
    _, best_metrics = context.evaluate(best_vec, record_series=True)
    if best_metrics is None:
        raise RuntimeError("Не удалось вычислить лучшую конфигурацию")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    best_params = {name: float(value) for (name, _, _), value in zip(PARAM_SPECS, best_vec)}
    save_json(out_dir / "best_params.json", best_params)
    metrics_for_json = {k: v for k, v in best_metrics.items() if k not in {"series", "measurements", "params", "timeseries"}}
    save_json(out_dir / "metrics.json", metrics_for_json)
    save_timeseries_csv(out_dir / "timeseries.csv", best_metrics)
    render_plots(out_dir, best_metrics)

    print("Лучшее значение целевой функции:", best_metrics["loss"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
