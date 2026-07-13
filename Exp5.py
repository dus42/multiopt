# %%
"""
Roundabout super-conflict benchmark
====================================

Classic N-aircraft "circle"/"roundabout" conflict: N aircraft are placed
evenly around a circle and each flies a straight track through the centre to
the diametrically opposite point. With equal start times every track reaches
the centre simultaneously, so all N(N-1)/2 pairs are in conflict at once - the
hardest stress test for a separation-constrained formulation.

Pipeline (mirrors the crossing-angle experiment):
  1. solve each aircraft independently (conflict=False) -> unconstrained floor
  2. build a vertically-layered warm start from those solutions (breaks the
     N-fold symmetry so IPOPT does not sit on a degenerate point)
  3. solve all aircraft jointly with the separation constraint, warm-started
  4. verify separation post-hoc over ALL pairs on an independent fine grid
  5. report fuel/penalty, per-pair residual conflicts, achieved separation,
     resolution efforts, and solver cost; draw the top-down resolution figure.

Expects the modified cruise.py (smooth interpolate_state_global +
candidate_sync_times warm-start path that accepts a per-scenario list).

Outputs:
  roundabout_results_summary.csv   (one-row run summary)
  roundabout_results_aircraft.csv  (per-aircraft metrics)
  roundabout_topdown.png           (free vs resolved tracks, top-down)
  roundabout_pairs.png             (per-pair min-ellipsoid: free vs resolved)
  roundabout_profiles.png          (altitude profiles + min pairwise sep vs t)

Note: to reproduce the *2-D* roundabout (lateral-only circulation), call
optimizer.fix_cruise_altitude() before .trajectory() so vertical resolution is
disabled and aircraft must circulate around the centre. See ALLOW_VERTICAL.
"""

import time
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openap
from openap import top

warnings.filterwarnings("ignore")

NM = openap.aero.nm
FT = openap.aero.ft

CONFIG = dict(
    n_aircraft=6,
    actype="A320",
    circle_radius_km=200.0,  # radius of the circle -> route length = 2*radius
    center=(51.0, 7.0),  # conflict point (lat, lon)
    m0=0.80,  # equal mass -> identical optimal FL -> all pairs conflict
    tstart=0,  # equal start -> simultaneous arrival at centre
    warmstart_layer_ft=0.0,  # vertical spread seeded into the warm start
    allow_vertical=False,  # False -> fix_cruise_altitude() for a 2-D roundabout
    max_nodes=30,
    max_iterations=5000,
    Rxy=5 * NM,
    Rz=1000 * FT,
    posthoc_dt=5.0,
    out_prefix="roundabout",
    out_suffix="200_fix_alt",
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def make_roundabout_scenarios(cfg):
    """N aircraft on a circle, each flying through the centre to the far side."""
    lat_c, lon_c = cfg["center"]
    R = cfg["circle_radius_km"] * 1000.0
    n = cfg["n_aircraft"]

    scenarios = []
    for i in range(n):
        theta = 360.0 * i / n  # position angle on the circle
        o_lat, o_lon = openap.aero.latlon(lat_c, lon_c, R, theta)
        d_lat, d_lon = openap.aero.latlon(lat_c, lon_c, R, (theta + 180.0) % 360.0)
        scenarios.append(
            {
                "actype": cfg["actype"],
                "origin": (round(float(o_lat), 4), round(float(o_lon), 4)),
                "destination": (round(float(d_lat), 4), round(float(d_lon), 4)),
                "m0": cfg["m0"],
                "tstart": cfg["tstart"],
                "id": i,
            }
        )
    return scenarios


def _proj(lon, lat, lat0, lon0):
    """Local tangent-plane projection (metres) about (lat0, lon0)."""
    bearings = openap.aero.bearing(lat0, lon0, lat, lon) * np.pi / 180.0
    distances = openap.aero.distance(lat0, lon0, lat, lon)
    return distances * np.sin(bearings), distances * np.cos(bearings)


# ---------------------------------------------------------------------------
# Warm start: vertically layered so IPOPT does not start on the symmetric point
# ---------------------------------------------------------------------------
def layered_warmstart(df_s, spread_ft):
    n = len(df_s)
    out = []
    for i, df in enumerate(df_s):
        d = df.copy()
        offset = (i - (n - 1) / 2.0) * (spread_ft / max(n - 1, 1))
        d["altitude"] = d["altitude"] + offset
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Post-hoc pairwise separation
# ---------------------------------------------------------------------------
def pair_separation(df1, df2, cfg):
    Rxy, Rz, dt = cfg["Rxy"], cfg["Rz"], cfg["posthoc_dt"]
    lat0 = 0.5 * (df1.latitude.mean() + df2.latitude.mean())
    lon0 = 0.5 * (df1.longitude.mean() + df2.longitude.mean())
    x1, y1 = _proj(df1.longitude.values, df1.latitude.values, lat0, lon0)
    x2, y2 = _proj(df2.longitude.values, df2.latitude.values, lat0, lon0)
    h1, h2 = df1.altitude.values * FT, df2.altitude.values * FT
    t1, t2 = df1.ts.values, df2.ts.values
    t0, tend = max(t1.min(), t2.min()), min(t1.max(), t2.max())
    if tend <= t0:
        return dict(
            min_horiz_nm=np.inf,
            min_ellipsoid=np.inf,
            n_conflict=0,
            t_cpa=np.nan,
            tg=np.array([]),
            horiz_nm=np.array([]),
        )
    tg = np.arange(t0, tend, dt)
    X1, Y1, H1 = (np.interp(tg, t1, a) for a in (x1, y1, h1))
    X2, Y2, H2 = (np.interp(tg, t2, a) for a in (x2, y2, h2))
    dx, dy, dz = X1 - X2, Y1 - Y2, H1 - H2
    horiz = np.sqrt(dx**2 + dy**2)
    ell = (dx / Rxy) ** 2 + (dy / Rxy) ** 2 + (dz / Rz) ** 8
    i = int(np.argmin(ell))
    return dict(
        min_horiz_nm=float(horiz.min() / NM),
        min_ellipsoid=float(ell.min()),
        n_conflict=int((ell < 1.0).sum()),
        t_cpa=float(tg[i]),
        tg=tg,
        horiz_nm=horiz / NM,
    )


def all_pairs(dfs, cfg):
    n = len(dfs)
    M_ell = np.full((n, n), np.nan)
    M_conf = np.zeros((n, n), dtype=int)
    total_conf = 0
    global_min_ell = np.inf
    global_min_horiz = np.inf
    series = {}  # (i,j) -> (tg, horiz_nm) for the time plot
    for i, j in itertools.combinations(range(n), 2):
        s = pair_separation(dfs[i], dfs[j], cfg)
        M_ell[i, j] = M_ell[j, i] = s["min_ellipsoid"]
        M_conf[i, j] = M_conf[j, i] = s["n_conflict"]
        total_conf += s["n_conflict"]
        global_min_ell = min(global_min_ell, s["min_ellipsoid"])
        global_min_horiz = min(global_min_horiz, s["min_horiz_nm"])
        series[(i, j)] = (s["tg"], s["horiz_nm"])
    return dict(
        M_ell=M_ell,
        M_conf=M_conf,
        total_conflicts=total_conf,
        min_ellipsoid=global_min_ell,
        min_horiz_nm=global_min_horiz,
        series=series,
    )


# ---------------------------------------------------------------------------
# Per-aircraft effort
# ---------------------------------------------------------------------------
def _path_len_km(df):
    d = openap.aero.distance(
        df.latitude.values[:-1],
        df.longitude.values[:-1],
        df.latitude.values[1:],
        df.longitude.values[1:],
    )
    return float(np.sum(d)) / 1000.0


def _alt_dev_ft(df_s, df_m):
    s_s = np.linspace(0, 1, len(df_s))
    s_m = np.linspace(0, 1, len(df_m))
    a_m = np.interp(s_s, s_m, df_m.altitude.values)
    return float(np.max(np.abs(a_m - df_s.altitude.values)))


def fuel(df):
    return float(df.mass.iloc[0] - df.mass.iloc[-1])


# ---------------------------------------------------------------------------
# Solver drivers
# ---------------------------------------------------------------------------
def _stats(opt):
    try:
        s = opt.solver.stats()
    except Exception:
        return dict(success=None, iters=None, t_wall=None)
    return dict(
        success=s.get("success", None),
        iters=s.get("iter_count", None),
        t_wall=s.get("t_wall_total", s.get("t_proc_total", None)),
    )


def run(cfg):
    scenarios = make_roundabout_scenarios(cfg)

    print(f"Solving {cfg['n_aircraft']} aircraft independently (floor + warm start)...")
    opt_s = top.Cruise(scenarios=scenarios, conflict=False, max_nodes=cfg["max_nodes"])
    if not cfg["allow_vertical"]:
        opt_s.fix_cruise_altitude()
    t0 = time.perf_counter()
    df_s = opt_s.trajectory(objective="fuel")
    t_single = time.perf_counter() - t0

    print("Solving jointly with separation constraint (layered warm start)...")
    warm = layered_warmstart(df_s, cfg["warmstart_layer_ft"])
    opt_m = top.Cruise(
        scenarios=scenarios,
        conflict=True,
        debug=True,
        max_nodes=cfg["max_nodes"],
        max_iterations=cfg["max_iterations"],
    )
    if not cfg["allow_vertical"]:
        opt_m.fix_cruise_altitude()
    t0 = time.perf_counter()
    df_m = opt_m.trajectory(objective="fuel", initial_guess=warm)
    t_multi = time.perf_counter() - t0

    return (
        df_s,
        df_m,
        dict(
            t_single=t_single,
            t_multi=t_multi,
            **{f"multi_{k}": v for k, v in _stats(opt_m).items()},
        ),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_topdown(df_s, df_m, cfg):
    lat_c, lon_c = cfg["center"]
    cmap = plt.cm.turbo(np.linspace(0, 1, len(df_m)))
    fig, (axf, axr) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, dfs, ttl in [
        (axf, df_s, "Unconstrained (straight through centre)"),
        (axr, df_m, "Separation-constrained resolution"),
    ]:
        for i, df in enumerate(dfs):
            x, y = _proj(df.longitude.values, df.latitude.values, lat_c, lon_c)
            ax.plot(x / 1000, y / 1000, color=cmap[i], lw=2, label=f"AC{i}")
            ax.scatter(x[0] / 1000, y[0] / 1000, color=cmap[i], s=30, zorder=5)
        ax.scatter(0, 0, marker="x", c="k", s=80, zorder=6)
        # 5 nm separation disk at the centre for scale
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(
            (cfg["Rxy"] / 1000) * np.cos(th),
            (cfg["Rxy"] / 1000) * np.sin(th),
            "k:",
            lw=1,
        )
        ax.set_aspect("equal")
        ax.set_title(ttl)
        ax.set_xlabel("east (km)")
        ax.grid(alpha=0.3)
    axf.set_ylabel("north (km)")
    axf.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{cfg['n_aircraft']}-aircraft roundabout: top-down")
    fig.tight_layout()
    fig.savefig(f"{cfg['out_prefix']}_topdown_{cfg['out_suffix']}.png", dpi=160)
    plt.close(fig)


def plot_pairs(free, res, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, M, ttl in [
        (axes[0], free["M_ell"], "Unconstrained min ellipsoid"),
        (axes[1], res["M_ell"], "Resolved min ellipsoid"),
    ]:
        im = ax.imshow(np.clip(M, 0, 4), cmap="RdYlGn", vmin=0, vmax=4)
        ax.set_title(ttl)
        ax.set_xlabel("aircraft")
        ax.set_ylabel("aircraft")
        n = M.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    ax.text(
                        j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7
                    )
        fig.colorbar(im, ax=ax, shrink=0.8, label="min ellipsoid (1=bound)")
    fig.tight_layout()
    fig.savefig(f"{cfg['out_prefix']}_pairs_{cfg['out_suffix']}.png", dpi=160)
    plt.close(fig)


def plot_profiles(df_m, res, cfg):
    cmap = plt.cm.turbo(np.linspace(0, 1, len(df_m)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    for i, df in enumerate(df_m):
        ax1.plot(df.ts, df.altitude, color=cmap[i], label=f"AC{i}")
    ax1.set_ylabel("altitude (ft)")
    ax1.set_title("Resolved altitude profiles")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, ncol=3)

    for (i, j), (tg, horiz) in res["series"].items():
        if len(tg):
            ax2.plot(tg, horiz, lw=0.8, alpha=0.6)
    ax2.axhline(cfg["Rxy"] / NM, color="r", ls="--", lw=1.2, label="5 nm bound")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("pairwise horizontal sep (nm)")
    ax2.set_title("All pairwise horizontal separations")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{cfg['out_prefix']}_profiles_{cfg['out_suffix']}.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg=CONFIG):
    df_s, df_m, meta = run(cfg)

    free = all_pairs(df_s, cfg)
    res = all_pairs(df_m, cfg)

    single_sum = sum(fuel(d) for d in df_s)
    multi_sum = sum(fuel(d) for d in df_m)

    # per-aircraft table
    rows = []
    for i in range(len(df_m)):
        rows.append(
            dict(
                aircraft=i,
                fuel_single_kg=fuel(df_s[i]),
                fuel_multi_kg=fuel(df_m[i]),
                penalty_kg=fuel(df_m[i]) - fuel(df_s[i]),
                alt_dev_ft=_alt_dev_ft(df_s[i], df_m[i]),
                path_increase_km=_path_len_km(df_m[i]) - _path_len_km(df_s[i]),
                cruise_alt_ft=float(np.median(df_m[i].altitude)),
            )
        )
    ac_df = pd.DataFrame(rows)
    ac_df.to_csv(
        f"{cfg['out_prefix']}_results_aircraft_{cfg['out_suffix']}.csv", index=False
    )

    summary = dict(
        n_aircraft=cfg["n_aircraft"],
        single_sum_kg=single_sum,
        multi_sum_kg=multi_sum,
        penalty_kg=multi_sum - single_sum,
        penalty_pct=100.0 * (multi_sum - single_sum) / single_sum,
        free_total_conflicts=free["total_conflicts"],
        res_total_conflicts=res["total_conflicts"],
        free_min_ellipsoid=free["min_ellipsoid"],
        res_min_ellipsoid=res["min_ellipsoid"],
        res_min_horiz_nm=res["min_horiz_nm"],
        max_alt_dev_ft=float(ac_df.alt_dev_ft.max()),
        max_path_increase_km=float(ac_df.path_increase_km.max()),
        **meta,
    )
    pd.DataFrame([summary]).to_csv(
        f"{cfg['out_prefix']}_results_summary_{cfg['out_suffix']}.csv", index=False
    )

    plot_topdown(df_s, df_m, cfg)
    plot_pairs(free, res, cfg)
    plot_profiles(df_m, res, cfg)

    print("\n=== ROUNDABOUT SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print("\nper-aircraft:")
    with pd.option_context("display.width", 160):
        print(ac_df.to_string(index=False))
    print(
        f"\nsaved {cfg['out_prefix']}_results_summary.csv, _results_aircraft_{cfg['out_suffix']}.csv, "
        f"and _topdown/_pairs/_profiles.png"
    )
    pd.concat(df_s).to_csv(f"data/exp5/roundabout_single_{cfg['out_suffix']}.csv")
    pd.concat(df_m).to_csv(f"data/exp5/roundabout_multi_{cfg['out_suffix']}.csv")
    return df_s, df_m, summary, ac_df


# %%

if __name__ == "__main__":
    main()
# %%
