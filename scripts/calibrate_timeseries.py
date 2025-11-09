#!/usr/bin/env python3
"""Calibration pipeline for the 1D delayed coking model against plant data."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover - informative message for users
    raise SystemExit(
        "pandas is required to read Excel files. Please install it via 'pip install pandas openpyxl'."
    ) from exc

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for reporting plots. Install it via 'pip install matplotlib'.") from exc

try:
    from scipy.optimize import differential_evolution, minimize  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scipy is required for optimisation. Please install it via 'pip install scipy'."
    ) from exc

from src.geometry import Geometry
from src.params import Inlet, Materials, TimeSetup, Walls, WallEnergy, WallLayer, MixtureEnergy
from src.kinetics import VR3Kinetics
from src.solver_1d import Coking1DSolver, h_mix


WEIGHTS = {
    "out": 0.35,
    "top": 0.30,
    "bot": 0.30,
    "height": 0.05,
}

DT = 30.0  # seconds


PARAM_SPECS = [
    ("wall.outer.k", 1.0, 40.0),
    ("wall.outer.rho", 3000.0, 8000.0),
    ("wall.outer.cp", 400.0, 900.0),
    ("wall.outer.thickness", 0.01, 0.25),
    ("wall.outer.epsilon", 0.60, 0.95),
    ("wall.inner.k", 1.0, 40.0),
    ("wall.inner.rho", 3000.0, 8000.0),
    ("wall.inner.cp", 400.0, 900.0),
    ("wall.inner.thickness", 0.01, 0.25),
    ("wall.inner.epsilon", 0.60, 0.95),
    ("wall.h_amb", 5.0, 50.0),
    ("wall.T_amb_C", 0.0, 50.0),
    ("mix.lambda_eff", 0.10, 3.0),
    ("mix.cp_eff", 1500.0, 3500.0),
    ("mix.h0_mix", 20.0, 800.0),
    ("mix.alpha_mdot", 0.20, 0.90),
    ("mix.alpha_p", 0.0, 0.40),
    ("mix.mdot_ref", 0.10, 200.0),
    ("mix.p_ref", 0.10, 20.0),
    ("kin.A_dist_scale", 0.30, 3.0),
    ("kin.A_coke_scale", 0.30, 3.0),
    ("kin.dT1_K", -20.0, 20.0),
    ("kin.dT2_K", -20.0, 20.0),
    ("kin.phi_por", 0.30, 1.0),
    ("kin.dH_dist_J_kg", -3e5, 3e5),
    ("kin.dH_coke_J_kg", -3e5, 3e5),
]


def _match_column(columns: Sequence[str], names: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for alias in names:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _interp_series(time_src: np.ndarray, values: np.ndarray, time_target: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return np.full_like(time_target, np.nan, dtype=float)
    return np.interp(time_target, time_src[mask], values[mask])


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.count_nonzero(mask) == 0:
        return 1e6
    denom = np.maximum(np.abs(y_true[mask]), 1e-3)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom)) * 100.0)


@dataclass
class TimeSeriesInputs:
    time_grid_s: np.ndarray
    meas_time_s: np.ndarray
    flow_kg_s: np.ndarray
    pressure: np.ndarray
    T_in_C: np.ndarray
    T_out_grid_C: np.ndarray
    T_head_top_grid_C: np.ndarray
    T_head_bottom_grid_C: np.ndarray
    T_out_meas_C: np.ndarray
    T_head_top_meas_C: np.ndarray
    T_head_bottom_meas_C: np.ndarray
    rho_vr: float


@dataclass
class SimulationResult:
    time_s: np.ndarray
    T_out_model_C: np.ndarray
    T_shell_top_model_C: np.ndarray
    T_shell_bottom_model_C: np.ndarray
    H_bed_m: np.ndarray
    q_mix_W_m3: np.ndarray


def prepare_inputs(path: Path, t_start_h: float, t_end_h: float, rho_vr: float) -> TimeSeriesInputs:
    df = pd.read_excel(path)
    columns = list(df.columns)

    col_time_h = _match_column(columns, ["time_h", "t_h", "hours", "time"])
    col_time_s = _match_column(columns, ["time_s", "t_s", "seconds"])
    if col_time_s:
        time_s = df[col_time_s].astype(float).to_numpy()
        time_h = time_s / 3600.0
    elif col_time_h:
        time_h = df[col_time_h].astype(float).to_numpy()
        time_s = time_h * 3600.0
    else:
        raise ValueError("Excel file must contain a time column (time_h or time_s).")

    col_flow = _match_column(columns, ["flow_m3_h", "flow", "q_m3_h", "feed_flow"])
    if not col_flow:
        raise ValueError("Flow column (m3/h) not found in Excel data.")
    flow_m3_h = df[col_flow].astype(float).to_numpy()

    col_pressure = _match_column(columns, ["pressure", "pressure_kgf_cm2", "p_kgf_cm2", "p"])
    if not col_pressure:
        raise ValueError("Pressure column (kgf/cm²) not found in Excel data.")
    pressure = df[col_pressure].astype(float).to_numpy()

    col_t_in = _match_column(columns, ["t_in", "feed_temp", "temp_in", "tin"])
    col_t_out = _match_column(columns, ["t_out", "product_temp", "temp_out", "tout"])
    if not col_t_in or not col_t_out:
        raise ValueError("Temperature columns T_in/T_out not found in Excel data.")
    T_in_C = df[col_t_in].astype(float).to_numpy()
    T_out_C = df[col_t_out].astype(float).to_numpy()

    col_top = _match_column(columns, ["t_head_upper", "t_upper", "head_top", "t_top"])
    col_bottom = _match_column(columns, ["t_head_lower", "t_lower", "head_bottom", "t_bottom"])
    if not col_top or not col_bottom:
        raise ValueError("Head temperatures (upper/lower) not found in Excel data.")
    T_head_top_C = df[col_top].astype(float).to_numpy()
    T_head_bottom_C = df[col_bottom].astype(float).to_numpy()

    mask = (time_h >= t_start_h) & (time_h <= t_end_h)
    if not np.any(mask):
        raise ValueError("Time range filters out all data points.")

    time_h = time_h[mask]
    time_s = time_s[mask]
    flow_m3_h = flow_m3_h[mask]
    pressure = pressure[mask]
    T_in_C = T_in_C[mask]
    T_out_meas_C = T_out_C[mask]
    T_head_top_meas_C = T_head_top_C[mask]
    T_head_bottom_meas_C = T_head_bottom_C[mask]

    t_start_s = float(time_s[0])
    t_end_s = float(time_s[-1])
    time_grid_s = np.arange(t_start_s, t_end_s + DT, DT)

    flow_m3_s = np.clip(flow_m3_h, 0.0, None) / 3600.0
    flow_kg_s = flow_m3_s * rho_vr

    flow_interp = _interp_series(time_s, flow_kg_s, time_grid_s)
    pressure_interp = _interp_series(time_s, pressure, time_grid_s)
    T_in_interp = _interp_series(time_s, T_in_C, time_grid_s)
    T_out_grid = _interp_series(time_s, T_out_meas_C, time_grid_s)
    T_top_interp = _interp_series(time_s, T_head_top_meas_C, time_grid_s)
    T_bot_interp = _interp_series(time_s, T_head_bottom_meas_C, time_grid_s)

    return TimeSeriesInputs(
        time_grid_s=time_grid_s,
        meas_time_s=time_s,
        flow_kg_s=flow_interp,
        pressure=pressure_interp,
        T_in_C=T_in_interp,
        T_out_grid_C=T_out_grid,
        T_head_top_grid_C=T_top_interp,
        T_head_bottom_grid_C=T_bot_interp,
        T_out_meas_C=T_out_meas_C,
        T_head_top_meas_C=T_head_top_meas_C,
        T_head_bottom_meas_C=T_head_bottom_meas_C,
        rho_vr=float(rho_vr),
    )


def vector_to_configs(vec: np.ndarray) -> Tuple[WallEnergy, MixtureEnergy, VR3Kinetics, Dict[str, float]]:
    params: Dict[str, float] = {}
    for value, (name, _, _) in zip(vec, PARAM_SPECS):
        params[name] = float(value)

    outer = WallLayer(
        k=params["wall.outer.k"],
        rho=params["wall.outer.rho"],
        cp=params["wall.outer.cp"],
        thickness=params["wall.outer.thickness"],
        epsilon=params["wall.outer.epsilon"],
    )
    inner = WallLayer(
        k=params["wall.inner.k"],
        rho=params["wall.inner.rho"],
        cp=params["wall.inner.cp"],
        thickness=params["wall.inner.thickness"],
        epsilon=params["wall.inner.epsilon"],
    )
    wall_energy = WallEnergy(
        outer=outer,
        inner=inner,
        h_amb=params["wall.h_amb"],
        T_amb_C=params["wall.T_amb_C"],
        zones=3,
    )

    mix_energy = MixtureEnergy(
        lambda_eff=params["mix.lambda_eff"],
        cp_eff=params["mix.cp_eff"],
        h0_mix=params["mix.h0_mix"],
        alpha_mdot=params["mix.alpha_mdot"],
        alpha_p=params["mix.alpha_p"],
        mdot_ref=max(params["mix.mdot_ref"], 1e-3),
        p_ref=max(params["mix.p_ref"], 1e-3),
    )

    kin = VR3Kinetics(
        A_dist_scale=params["kin.A_dist_scale"],
        A_coke_scale=params["kin.A_coke_scale"],
        dT1_K=params["kin.dT1_K"],
        dT2_K=params["kin.dT2_K"],
        phi_por=min(max(params["kin.phi_por"], 0.3), 1.0),
        dH_dist_J_kg=params["kin.dH_dist_J_kg"],
        dH_coke_J_kg=params["kin.dH_coke_J_kg"],
    )

    return wall_energy, mix_energy, kin, params


def simulate(
    inputs: TimeSeriesInputs,
    wall_energy: WallEnergy,
    mix_energy: MixtureEnergy,
    kin: VR3Kinetics,
) -> Optional[SimulationResult]:
    time_rel = inputs.time_grid_s - inputs.time_grid_s[0]
    geom = Geometry(H=21.25, D=5.5, NZ=Geometry().NZ)
    inlet = Inlet(
        T_in_C=float(inputs.T_in_C[0]),
        m_dot_kg_s=float(np.clip(inputs.flow_kg_s[0], 1e-6, None)),
        rho_vr=inputs.rho_vr,
    )
    walls = Walls()
    mats = Materials()
    tcfg = TimeSetup(total_hours=time_rel[-1] / 3600.0, dt=DT, snapshots_h=())
    solver = Coking1DSolver(geom, inlet, walls, mats, tcfg, kin, wall_energy=wall_energy, mix_energy=mix_energy)

    time_hist: List[float] = []
    T_out_hist: List[float] = []
    T_top_hist: List[float] = []
    T_bot_hist: List[float] = []
    H_hist: List[float] = []
    q_hist: List[float] = []

    time_hist.append(0.0)
    T_out_hist.append(float(solver.T[-1]))
    T_top_hist.append(float(solver.T_shell_out_K[2] - 273.15))
    T_bot_hist.append(float(solver.T_shell_out_K[0] - 273.15))
    H_hist.append(float(solver.bed_height_equiv()))
    q_hist.append(0.0)

    for i in range(1, len(time_rel)):
        solver.inlet.T_in_C = float(inputs.T_in_C[i])
        solver.inlet.m_dot_kg_s = float(np.clip(inputs.flow_kg_s[i], 1e-6, None))
        pressure = float(np.clip(inputs.pressure[i], 1e-3, None))

        top_obs = (
            float(inputs.T_head_top_grid_C[i]) if np.isfinite(inputs.T_head_top_grid_C[i]) else None
        )
        bot_obs = (
            float(inputs.T_head_bottom_grid_C[i]) if np.isfinite(inputs.T_head_bottom_grid_C[i]) else None
        )
        solver.set_shell_observations(time_rel[i], top_obs, bot_obs)

        solver.step(pressure=pressure)

        if not (np.all(np.isfinite(solver.T_mix_K)) and np.all(np.isfinite(solver.aR))):
            return None

        time_hist.append(time_rel[i])
        T_out_hist.append(float(solver.T[-1]))
        T_top_hist.append(float(solver.T_shell_out_K[2] - 273.15))
        T_bot_hist.append(float(solver.T_shell_out_K[0] - 273.15))
        H_hist.append(float(solver.bed_height_equiv()))
        q_hist.append(float(h_mix(solver.inlet.m_dot_kg_s, pressure, mix_energy)))

    return SimulationResult(
        time_s=np.asarray(time_hist, dtype=float),
        T_out_model_C=np.asarray(T_out_hist, dtype=float),
        T_shell_top_model_C=np.asarray(T_top_hist, dtype=float),
        T_shell_bottom_model_C=np.asarray(T_bot_hist, dtype=float),
        H_bed_m=np.asarray(H_hist, dtype=float),
        q_mix_W_m3=np.asarray(q_hist, dtype=float),
    )


def compute_metrics(inputs: TimeSeriesInputs, sim: SimulationResult) -> Dict[str, float]:
    meas_time_rel = inputs.meas_time_s - inputs.time_grid_s[0]
    model_out = _interp_series(sim.time_s, sim.T_out_model_C, meas_time_rel)
    model_top = _interp_series(sim.time_s, sim.T_shell_top_model_C, meas_time_rel)
    model_bottom = _interp_series(sim.time_s, sim.T_shell_bottom_model_C, meas_time_rel)

    meas_out = inputs.T_out_meas_C
    meas_top = inputs.T_head_top_meas_C
    meas_bottom = inputs.T_head_bottom_meas_C

    mape_out = _mape(meas_out, model_out)
    mape_top = _mape(meas_top, model_top)
    mape_bottom = _mape(meas_bottom, model_bottom)

    H_final = float(sim.H_bed_m[-1])
    metrics = {
        "MAPE_out": mape_out,
        "MAPE_top": mape_top,
        "MAPE_bottom": mape_bottom,
        "MAPE_avg": (mape_out + mape_top + mape_bottom) / 3.0,
        "H_final_m": H_final,
    }
    return metrics


def loss_function(metrics: Dict[str, float]) -> float:
    penalty_height = abs(metrics["H_final_m"] - 17.5) / 17.5
    return (
        WEIGHTS["out"] * metrics["MAPE_out"]
        + WEIGHTS["top"] * metrics["MAPE_top"]
        + WEIGHTS["bot"] * metrics["MAPE_bottom"]
        + WEIGHTS["height"] * penalty_height * 100.0
    )


def optimise(inputs: TimeSeriesInputs, rng_seed: int, max_iter: int, pop_size: int, use_lbfgs: bool):
    bounds = [(lo, hi) for _, lo, hi in PARAM_SPECS]

    def objective(vec: np.ndarray) -> float:
        wall_energy, mix_energy, kin, _ = vector_to_configs(vec)
        sim = simulate(inputs, wall_energy, mix_energy, kin)
        if sim is None:
            return 1e6
        metrics = compute_metrics(inputs, sim)
        value = loss_function(metrics)
        if not math.isfinite(value):
            return 1e6
        return value

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max_iter,
        popsize=pop_size,
        seed=rng_seed,
        polish=False,
        tol=1e-3,
    )

    x_best = result.x
    if use_lbfgs:
        res_local = minimize(
            objective,
            x_best,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-6},
        )
        if res_local.success:
            x_best = res_local.x

    wall_energy, mix_energy, kin, params = vector_to_configs(x_best)
    sim = simulate(inputs, wall_energy, mix_energy, kin)
    if sim is None:
        raise RuntimeError("Simulation failed for best parameters")
    metrics = compute_metrics(inputs, sim)
    loss_val = loss_function(metrics)
    return params, sim, metrics, loss_val


def export_reports(
    outdir: Path,
    params: Dict[str, float],
    sim: SimulationResult,
    inputs: TimeSeriesInputs,
    metrics: Dict[str, float],
    loss_val: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    best_params_path = outdir / "best_params.json"
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    metrics_dict = metrics.copy()
    metrics_dict["loss"] = loss_val
    metrics_dict["weights"] = WEIGHTS
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    time_rel = inputs.time_grid_s - inputs.time_grid_s[0]
    headers = [
        "time_s",
        "time_h",
        "T_out_model_C",
        "T_out_meas_C",
        "T_shell_top_model_C",
        "T_shell_top_meas_C",
        "T_shell_bottom_model_C",
        "T_shell_bottom_meas_C",
        "H_bed_model_m",
    ]

    meas_top = inputs.T_head_top_grid_C
    meas_bottom = inputs.T_head_bottom_grid_C
    meas_out = inputs.T_out_grid_C

    with (outdir / "timeseries.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for t, model_out, model_top, model_bottom, H in zip(
            time_rel,
            sim.T_out_model_C,
            sim.T_shell_top_model_C,
            sim.T_shell_bottom_model_C,
            sim.H_bed_m,
        ):
            idx = np.searchsorted(time_rel, t)
            idx = min(idx, len(time_rel) - 1)
            row = [
                f"{t:.1f}",
                f"{t/3600.0:.4f}",
                f"{model_out:.3f}",
                f"{meas_out[idx]:.3f}" if np.isfinite(meas_out[idx]) else "",
                f"{model_top:.3f}",
                f"{meas_top[idx]:.3f}" if np.isfinite(meas_top[idx]) else "",
                f"{model_bottom:.3f}",
                f"{meas_bottom[idx]:.3f}" if np.isfinite(meas_bottom[idx]) else "",
                f"{H:.4f}",
            ]
            f.write(",".join(row) + "\n")

    t_h = sim.time_s / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t_h, sim.T_out_model_C, label="Модель", lw=2.0)
    axes[0].plot(time_rel / 3600.0, meas_out, "o", label="Измерение", ms=3.0, alpha=0.7)
    axes[0].set_ylabel("T, °C")
    axes[0].set_title("Температура продукта")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t_h, sim.T_shell_top_model_C, lw=2.0)
    axes[1].plot(time_rel / 3600.0, meas_top, "o", ms=3.0, alpha=0.7)
    axes[1].set_ylabel("T, °C")
    axes[1].set_title("Оболочка — верхнее днище")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_h, sim.T_shell_bottom_model_C, lw=2.0)
    axes[2].plot(time_rel / 3600.0, meas_bottom, "o", ms=3.0, alpha=0.7)
    axes[2].set_ylabel("T, °C")
    axes[2].set_xlabel("Время, ч")
    axes[2].set_title("Оболочка — нижнее днище")
    axes[2].grid(True, alpha=0.3)
    fig.savefig(outdir / "comparison_temperatures.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(t_h, sim.H_bed_m * 100.0, label="Модель", lw=2.0)
    ax.axhline(17.0 * 100.0, color="g", linestyle="--", label="Целевой диапазон")
    ax.axhline(18.0 * 100.0, color="g", linestyle="--")
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Высота, см")
    ax.set_title("Рост коксового слоя")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(outdir / "comparison_height.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate 1D coking model against time-series measurements.")
    parser.add_argument("--excel", type=Path, required=True, help="Path to Excel file with plant data.")
    parser.add_argument("--t-start", type=float, required=True, help="Start time (h).")
    parser.add_argument("--t-end", type=float, required=True, help="End time (h).")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="Output directory for calibration artefacts.")
    parser.add_argument("--rho", type=float, default=1050.0, help="Feed density, kg/m³ (for flow conversion).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for differential evolution.")
    parser.add_argument("--max-iter", type=int, default=40, help="Maximum iterations for differential evolution.")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size for differential evolution.")
    parser.add_argument("--no-lbfgs", action="store_true", help="Disable local L-BFGS-B polishing step.")

    args = parser.parse_args()

    inputs = prepare_inputs(args.excel, args.t_start, args.t_end, args.rho)
    params, sim, metrics, loss_val = optimise(
        inputs,
        rng_seed=args.seed,
        max_iter=args.max_iter,
        pop_size=args.pop_size,
        use_lbfgs=not args.no_lbfgs,
    )

    export_reports(args.out, params, sim, inputs, metrics, loss_val)

    print("Calibration finished.")
    print(f"Loss: {loss_val:.4f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()

