# src/visualization.py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def render_all_ru(results: dict, outdir: Path, exp_h_cm: float = None, exp_y_pct: float = None):
    """Экспорт результатов (укрупнённые картинки; профили 1×3, карты 1×4)."""
    outdir.mkdir(parents=True, exist_ok=True)

    z_m  = np.asarray(results['z'])
    z_cm = z_m * 100.0
    snaps = results.get('snapshots', {})
    cont  = results.get('contours', {})
    meta  = results.get('meta', {})

    times_h = snaps.get('t_h', [])
    step = max(1, len(times_h) // 6) if len(times_h) > 0 else 1
    lw = 2.0
    dpi = 200

    # Общие размеры шрифтов для этого экспорта
    rc = {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }

    import matplotlib.pyplot as plt
    from matplotlib import cm

    with plt.rc_context(rc):
        # 1) ПРОФИЛИ (1×3)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
        if len(times_h) > 0:
            for ax, key, title in zip(axes, ['aR', 'aD', 'aC'], ['VR', 'Дистилляты', 'Кокс']):
                arr = np.asarray(snaps.get(key, []))
                if arr.size:
                    for j in range(0, len(times_h), step):
                        ax.plot(arr[j], z_cm, lw=lw, label=f'{times_h[j]:.1f} ч')
                ax.set_title(title)
                ax.set_xlabel('Доля объёма')
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel('Высота (см)')
            axes[0].legend(loc='upper left', ncol=3)
        fig.savefig(outdir / 'profiles.png', dpi=dpi)
        plt.close(fig)

        # 2) ТЕМПЕРАТУРА (крупнее)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        if len(times_h) > 0 and np.asarray(snaps.get('T', [])).size:
            Tarr = np.asarray(snaps['T'])
            for j in range(0, len(times_h), step):
                ax.plot(Tarr[j], z_cm, lw=lw, label=f'{times_h[j]:.1f} ч')
            ax.legend(loc='best')
        ax.set_xlabel('T (°C)')
        ax.set_ylabel('Высота (см)')
        ax.set_title('Эволюция температурного профиля')
        ax.grid(True, alpha=0.3)
        fig.savefig(outdir / 'temperature.png', dpi=dpi)
        plt.close(fig)

        # 3) ПОРИСТОСТЬ (крупнее)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        if len(times_h) > 0 and np.asarray(snaps.get('aC', [])).size:
            Carr = np.asarray(snaps['aC'])
            for j in range(0, len(times_h), step):
                ax.plot(1.0 - Carr[j], z_cm, lw=lw, label=f'{times_h[j]:.1f} ч')
            ax.legend(loc='best')
        ax.set_xlabel('Пористость γ')
        ax.set_ylabel('Высота (см)')
        ax.set_xlim(0, 1)
        ax.set_title('Эволюция пористости (γ = 1 − aC)')
        ax.grid(True, alpha=0.3)
        fig.savefig(outdir / 'porosity.png', dpi=dpi)
        plt.close(fig)

        # 4) РОСТ СЛОЯ
        growth = results.get('growth', {})
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax.plot(growth.get('t_h', []), growth.get('H_cm', []), 'b-', lw=lw, label='Эквивалентная')
        thr = float(results.get('extra', {}).get('front_thr', 0.10))
        if len(cont.get('t_s', [])) > 0 and np.asarray(cont.get('aC', [])).size:
            t_h = np.asarray(cont['t_s']) / 3600.0
            aC_hist = np.asarray(cont['aC'])
            dz = float(z_m[1] - z_m[0]) if z_m.size > 1 else 0.0
            H_front = []
            for aC in aC_hist:
                idx = np.where(aC > thr)[0]
                H = ((idx[-1] + 1) * dz if idx.size else 0.0) * 100.0
                H_front.append(H)
            ax.plot(t_h, H_front, 'r--', lw=1.7, label=f'Фронт ({int(thr*100)}%)')
        ax.set_xlabel('Время (ч)')
        ax.set_ylabel('Высота (см)')
        ax.set_title('Рост коксового слоя')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(outdir / 'growth.png', dpi=dpi)
        plt.close(fig)

        # 4b) СРАВНЕНИЕ С ИЗМЕРЕНИЯМИ (если доступны)
        ts = results.get('timeseries')
        if ts:
            t_s = np.asarray(ts.get('t_s', []), dtype=float)
            if t_s.size:
                t_h = t_s / 3600.0
                fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
                data_pairs = [
                    ('T_out_model_C', 'T_out_meas_C', 'Выход'),
                    ('T_upper_model_C', 'T_upper_meas_C', 'Верхнее днище'),
                    ('T_lower_model_C', 'T_lower_meas_C', 'Нижнее днище'),
                ]
                for ax_i, (model_key, meas_key, title) in zip(axes, data_pairs):
                    model = np.asarray(ts.get(model_key, []), dtype=float)
                    meas = np.asarray(ts.get(meas_key, []), dtype=float)
                    ax_i.plot(t_h[:model.size], model, lw=lw, label='Модель')
                    if meas.size:
                        mask = np.isfinite(meas)
                        ax_i.scatter(t_h[:meas.size][mask], meas[mask], c='k', s=18, label='Измерения')
                    ax_i.set_ylabel('T, °C')
                    ax_i.set_title(title)
                    ax_i.grid(True, alpha=0.3)
                axes[-1].set_xlabel('Время (ч)')
                axes[0].legend()
                fig.savefig(outdir / 'timeseries_compare.png', dpi=dpi)
                plt.close(fig)

                H = np.asarray(ts.get('H_m', []), dtype=float)
                if H.size:
                    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                    ax.plot(t_h[:H.size], H, lw=lw)
                    ax.set_xlabel('Время (ч)')
                    ax.set_ylabel('Высота слоя (м)')
                    ax.set_title('Высота слоя во времени (модель)')
                    ax.grid(True, alpha=0.3)
                    fig.savefig(outdir / 'timeseries_height.png', dpi=dpi)
                    plt.close(fig)

        # 5) ВЫХОДЫ ФАЗ (серия) + 6) ФИНАЛ
        if len(cont.get('t_s', [])) > 0:
            A = float(meta['A_m2']); m_dot = float(meta['m_dot_kg_s'])
            rho_vr = float(meta['rho_vr']); rho_coke = float(meta['rho_coke'])
            t_s = np.asarray(cont['t_s'])
            aR = np.asarray(cont['aR']); aC = np.asarray(cont['aC'])
            m_vr = A * np.trapz(aR * rho_vr, z_m, axis=1)
            m_coke = A * np.trapz(aC * rho_coke, z_m, axis=1)
            m_vr0 = float(meta.get('m_vr0', m_vr[0] if m_vr.size else 0.0))
            M_feed = m_dot * t_s
            M_total = np.maximum(M_feed + m_vr0, 1e-12)
            Y_vr = 100.0 * m_vr / M_total
            Y_ck = 100.0 * m_coke / M_total
            Y_ds = 100.0 - Y_vr - Y_ck
            t_h = t_s / 3600.0

            fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            ax.plot(t_h, Y_vr, lw=lw, label='VR (остаток)')
            ax.plot(t_h, Y_ds, lw=lw, label='Дистилляты (накоплено)')
            ax.plot(t_h, Y_ck, lw=lw, label='Кокс')
            ax.set_xlabel('Время (ч)')
            ax.set_ylabel('Выход, % (баланс масс)')
            ax.set_title('Выходы фаз во времени')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(outdir / 'phase_yields.png', dpi=dpi)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            vals = [float(Y_vr[-1]), float(Y_ds[-1]), float(Y_ck[-1])]
            bars = ax.bar(['VR', 'Дистилляты', 'Кокс'], vals)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.6, f'{v:.2f}%', ha='center')
            ax.set_ylabel('% (баланс масс)')
            ax.set_title('Финальные выходы фаз')
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.2)
            fig.savefig(outdir / 'final_yields.png', dpi=dpi)
            plt.close(fig)

        # 7) ТЕПЛОВЫЕ КАРТЫ (1×4)
        if len(cont.get('t_s', [])) > 0:
            t_h = np.asarray(cont['t_s']) / 3600.0
            extent = [t_h.min(), t_h.max(), z_cm.min(), z_cm.max()]
            data_list = [np.asarray(cont['aR']), np.asarray(cont['aD']),
                         np.asarray(cont['aC']), np.asarray(cont['T'])]
            titles = ['VR', 'Дистилляты', 'Кокс', 'T (°C)']
            fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True, sharey=True)
            for ax, data, title in zip(axes, data_list, titles):
                im = ax.imshow(data.T, origin='lower', aspect='auto', extent=extent, cmap='viridis')
                ax.set_title(title)
                ax.set_xlabel('Время (ч)')
                if ax is axes[0]:
                    ax.set_ylabel('Высота (см)')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.savefig(outdir / 'heatmaps.png', dpi=dpi)
            plt.close(fig)

        # 8) Текстовый отчёт (оставляем как было)
        with open(outdir / 'results.txt', 'w', encoding='utf-8') as f:
            H_cm = results['H_bed_m'] * 100.0
            H_front_cm = results['H_front_m'] * 100.0
            Y_feed = results.get('yield_pct', 0.0)
            T_avg = results.get('T_avg_C', 0.0)
            por_avg = results.get('porosity_avg', 0.0)

            final = results.get('final', {})
            A = float(meta.get('A_m2', 0.0))
            m_dot = float(meta.get('m_dot_kg_s', 0.0))
            rho_vr = float(meta.get('rho_vr', 0.0))
            rho_c = float(meta.get('rho_coke', 0.0))

            if len(cont.get('t_s', [])) > 0:
                t_end_s = float(cont['t_s'][-1])
            elif len(results.get('growth', {}).get('t_h', [])) > 0:
                t_end_s = float(results['growth']['t_h'][-1] * 3600.0)
            else:
                t_end_s = 0.0

            m_vr0 = float(meta.get('m_vr0', 0.0))
            aR_fin = np.asarray(final.get('aR', []))
            aC_fin = np.asarray(final.get('aC', []))
            M_feed = m_dot * t_end_s
            M_vr_end = float(A * np.trapz(aR_fin * rho_vr, z_m)) if aR_fin.size else 0.0
            M_coke   = float(A * np.trapz(aC_fin * rho_c,  z_m)) if aC_fin.size else 0.0
            M_dist   = max(M_feed + m_vr0 - M_vr_end - M_coke, 0.0)
            Y_c_bal  = 100.0 * M_coke / max(M_feed + m_vr0, 1e-12)

            f.write("="*60 + "\n")
            f.write("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ\n")
            f.write("="*60 + "\n\n")
            f.write("Финальные показатели:\n")
            f.write(f"  Высота слоя (экв.):  {H_cm:.2f} см\n")
            f.write(f"  Высота фронта:       {H_front_cm:.2f} см\n")
            f.write(f"  Выход кокса (по подаче): {Y_feed:.2f} %\n")
            f.write(f"  Средняя температура: {T_avg:.1f} °C\n")
            f.write(f"  Средняя пористость:  {por_avg:.3f}\n")
            f.write(f"  Балансный выход кокса: {Y_c_bal:.2f} %\n\n")
            f.write("Массовый баланс (кг):\n")
            f.write(f"  VR(начало)   = {m_vr0:.3f}\n")
            f.write(f"  Подача       = {M_feed:.3f}\n")
            f.write(f"  VR(конец)    = {M_vr_end:.3f}\n")
            f.write(f"  Кокс         = {M_coke:.3f}\n")
            f.write(f"  Дистилляты   = {M_dist:.3f}\n")

            extra = results.get('extra', {})
            if extra.get('t_fill_h') is not None:
                f.write("\nМомент заполнения барабана:\n")
                f.write(f"  t_fill = {extra['t_fill_h']:.2f} ч\n")
                f.write(f"  Выход к t_fill:      {extra['Y_at_fill_pct']:.2f} % ")
                f.write(f"(порог aC>{extra.get('front_thr', 0.10)*100:.0f}%)\n")

            if final and len(z_cm) > 0:
                f.write("\nФинальный профиль (выборочные точки):\n")
                f.write(f"{'z, см':>8} {'T, °C':>8} {'α_VR':>8} {'α_coke':>8} {'ε':>8}\n")
                f.write("-"*44 + "\n")
                idxs = [0, len(z_cm)//4, len(z_cm)//2, 3*len(z_cm)//4, len(z_cm)-1]
                for i in idxs:
                    eps = 1.0 - float(final['aC'][i])
                    f.write(f"{z_cm[i]:>8.1f} {float(final['T'][i]):>8.1f} "
                            f"{float(final['aR'][i]):>8.3f} {float(final['aC'][i]):>8.3f} "
                            f"{eps:>8.3f}\n")
