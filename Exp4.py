# %%
"""
Crossing-angle sweep experiment
================================

Headline experiment for the conference paper on separation-constrained
trajectory optimisation (OpenAP.top + CasADi + IPOPT, direct collocation).

For each crossing angle theta in a sweep, two aircraft are placed on straight
tracks that intersect at a common point at (approximately) the same time, so a
genuine conflict is induced. For each angle we:

  1. solve each aircraft independently (conflict=False)  -> unconstrained floor
                                                           + warm start
  2. solve both jointly with the separation constraint (conflict=True),
     warm-started from step 1
  3. verify separation post-hoc on an independent fine time grid
  4. record fuel, fuel penalty, achieved separation, residual conflicts,
     how the conflict was resolved (vertical / lateral / temporal effort),
     and solver cost.

Outputs:
  - crossing_sweep_results.csv   (one row per angle)
  - crossing_sweep_penalty.png   (fuel penalty + resolution mode vs angle)
  - crossing_sweep_efforts.png   (resolution efforts + achieved separation)

This script expects the modified cruise.py (smooth interpolate_state_global,
candidate_sync_times warm-start path). It does not modify the solver; it only
drives it and analyses the output.
"""

import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openap
from openap import top

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NM = openap.aero.nm
FT = openap.aero.ft

CONFIG = dict(
    angles_deg=np.arange(30, 151, 15),  # 30, 45, ..., 150
    actype="A320",  # same type both aircraft (NEEDED focus)
    route_km=600.0,  # total track length per aircraft
    center=(51.0, 7.0),  # crossing point (lat, lon), central Europe
    m0=(0.80, 0.81),  # slight mass diff breaks the up/down tie
    tstart=(0, 0),  # equal start -> simultaneous at crossing
    max_nodes=40,
    max_iterations=3000,
    Rxy=5 * NM,  # horizontal separation radius
    Rz=1000 * FT,  # vertical separation radius
    posthoc_dt=5.0,  # post-hoc verification grid spacing [s]
    out_prefix="crossing_sweep",
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def make_crossing_scenario(angle_deg, cfg):
    """Two straight tracks crossing at `center` with the given angle between
    their headings, each of length route_km, centred on the crossing point."""
    lat_c, lon_c = cfg["center"]
    half = cfg["route_km"] * 1000.0 / 2.0
    # symmetric about due-east so every angle is centred the same way
    b1 = 90.0 - angle_deg / 2.0
    b2 = 90.0 + angle_deg / 2.0

    scenarios = []
    for k, (b, m, ts) in enumerate(zip([b1, b2], cfg["m0"], cfg["tstart"])):
        o_lat, o_lon = openap.aero.latlon(lat_c, lon_c, half, (b + 180.0) % 360.0)
        d_lat, d_lon = openap.aero.latlon(lat_c, lon_c, half, b)
        scenarios.append(
            {
                "actype": cfg["actype"],
                "origin": (round(float(o_lat), 4), round(float(o_lon), 4)),
                "destination": (round(float(d_lat), 4), round(float(d_lon), 4)),
                "m0": m,
                "tstart": ts,
                "id": k,
            }
        )
    return scenarios


def _proj(lon, lat, lat0, lon0):
    """Local tangent-plane projection (metres) about (lat0, lon0)."""
    bearings = openap.aero.bearing(lat0, lon0, lat, lon) * np.pi / 180.0
    distances = openap.aero.distance(lat0, lon0, lat, lon)
    return distances * np.sin(bearings), distances * np.cos(bearings)


# ---------------------------------------------------------------------------
# Post-hoc separation check (independent of the solver's own grid)
# ---------------------------------------------------------------------------
def compute_separation(df1, df2, cfg):
    """Interpolate both trajectories onto a common fine time grid over their
    temporal overlap and report the binding separation metrics."""
    Rxy, Rz, dt = cfg["Rxy"], cfg["Rz"], cfg["posthoc_dt"]

    lat0 = 0.5 * (df1.latitude.mean() + df2.latitude.mean())
    lon0 = 0.5 * (df1.longitude.mean() + df2.longitude.mean())

    x1, y1 = _proj(df1.longitude.values, df1.latitude.values, lat0, lon0)
    x2, y2 = _proj(df2.longitude.values, df2.latitude.values, lat0, lon0)
    h1 = df1.altitude.values * FT
    h2 = df2.altitude.values * FT
    t1, t2 = df1.ts.values, df2.ts.values

    t0 = max(t1.min(), t2.min())
    tend = min(t1.max(), t2.max())
    if tend <= t0:
        return dict(
            min_horiz_nm=np.inf,
            vert_at_cpa_ft=np.inf,
            min_ellipsoid=np.inf,
            n_conflict=0,
            t_cpa=np.nan,
            vert_share=np.nan,
        )

    tg = np.arange(t0, tend, dt)
    X1, Y1, H1 = (np.interp(tg, t1, a) for a in (x1, y1, h1))
    X2, Y2, H2 = (np.interp(tg, t2, a) for a in (x2, y2, h2))

    dx, dy, dz = X1 - X2, Y1 - Y2, H1 - H2
    horiz = np.sqrt(dx**2 + dy**2)
    ch = (dx**2 + dy**2) / Rxy**2
    cz = (dz / Rz) ** 8
    ell = ch + cz

    i = int(np.argmin(ell))  # binding point (closest approach in ellipsoid sense)
    vert_share = cz[i] / (cz[i] + ch[i] + 1e-12)

    return dict(
        min_horiz_nm=float(horiz.min() / NM),
        vert_at_cpa_ft=float(abs(dz[i]) / FT),
        min_ellipsoid=float(ell.min()),
        n_conflict=int((ell < 1.0).sum()),
        t_cpa=float(tg[i]),
        vert_share=float(vert_share),  # share of the ellipsoid met vertically at CPA
    )


# ---------------------------------------------------------------------------
# Resolution-effort metrics (how the conflict was resolved, vs the floor)
# ---------------------------------------------------------------------------
def _path_length_m(df):
    d = openap.aero.distance(
        df.latitude.values[:-1],
        df.longitude.values[:-1],
        df.latitude.values[1:],
        df.longitude.values[1:],
    )
    return float(np.sum(d))


def _alt_dev_ft(df_single, df_multi):
    """Max altitude deviation of the constrained track from its own free track,
    matched by along-route progress fraction."""
    s_s = np.linspace(0, 1, len(df_single))
    s_m = np.linspace(0, 1, len(df_multi))
    a_m = np.interp(s_s, s_m, df_multi.altitude.values)
    return float(np.max(np.abs(a_m - df_single.altitude.values)))


def resolution_efforts(df_s, df_m):
    """Per-aircraft effort relative to the unconstrained solution."""
    vert_ft = max(_alt_dev_ft(df_s[k], df_m[k]) for k in range(len(df_s)))
    lat_km = max(
        (_path_length_m(df_m[k]) - _path_length_m(df_s[k])) / 1000.0
        for k in range(len(df_s))
    )
    dur = lambda d: float(d.ts.iloc[-1] - d.ts.iloc[0])
    temp_s = max(abs(dur(df_m[k]) - dur(df_s[k])) for k in range(len(df_s)))
    return dict(vert_ft=vert_ft, lat_km=lat_km, temp_s=temp_s)


def classify_mode(sep, efforts, cfg):
    """Heuristic label for the dominant resolution mode at the binding point.
    Documented as a heuristic in the paper, not a hard taxonomy."""
    vs = sep["vert_share"]
    # significant timing change (relative to a spatial-equivalent threshold)
    temporal_used = efforts["temp_s"] > 30.0
    if np.isnan(vs):
        return "none"
    if vs > 0.6:
        mode = "vertical"
    elif vs < 0.4:
        mode = "lateral"
    else:
        mode = "mixed"
    if temporal_used:
        mode += "+temporal"
    return mode


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


def run_pair(scenarios, cfg, angle):
    """Single (floor + warm start) then constrained (warm-started) solve."""
    opt_s = top.Cruise(scenarios=scenarios, conflict=False, max_nodes=cfg["max_nodes"])
    t0 = time.perf_counter()
    df_s = opt_s.trajectory(objective="fuel")
    t_single = time.perf_counter() - t0
    st_s = _stats(opt_s)

    opt_m = top.Cruise(
        scenarios=scenarios,
        conflict=True,
        max_nodes=cfg["max_nodes"],
        max_iterations=cfg["max_iterations"],
    )
    t0 = time.perf_counter()
    df_m = opt_m.trajectory(objective="fuel", initial_guess=df_s)
    t_multi = time.perf_counter() - t0
    st_m = _stats(opt_m)
    pd.concat(
        [pd.concat(df_s).assign(sm="single"), pd.concat(df_m).assign(sm="multi")]
    ).to_csv(f"data/exp4/{angle}.csv")
    return (
        df_s,
        df_m,
        dict(
            t_single=t_single,
            t_multi=t_multi,
            single_success=st_s["success"],
            multi_success=st_m["success"],
            multi_iters=st_m["iters"],
            multi_t_wall=st_m["t_wall"],
        ),
    )


def fuel(df):
    return float(df.mass.iloc[0] - df.mass.iloc[-1])


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(cfg):
    rows = []
    for angle in cfg["angles_deg"]:
        print(f"\n=== crossing angle {angle:.0f} deg ===")
        scenarios = make_crossing_scenario(angle, cfg)
        try:
            df_s, df_m, meta = run_pair(scenarios, cfg, angle)
        except Exception as e:  # keep the sweep going if one geometry fails
            print(f"  FAILED: {e}")
            rows.append(dict(angle_deg=angle, error=str(e)))
            continue

        single_sum = sum(fuel(d) for d in df_s)
        multi_sum = sum(fuel(d) for d in df_m)
        penalty_pct = 100.0 * (multi_sum - single_sum) / single_sum

        sep_free = compute_separation(df_s[0], df_s[1], cfg)  # was there a conflict?
        sep_res = compute_separation(df_m[0], df_m[1], cfg)  # resolved?
        efforts = resolution_efforts(df_s, df_m)
        mode = classify_mode(sep_res, efforts, cfg)

        row = dict(
            angle_deg=float(angle),
            single_sum_kg=single_sum,
            multi_sum_kg=multi_sum,
            penalty_kg=multi_sum - single_sum,
            penalty_pct=penalty_pct,
            free_min_ellipsoid=sep_free["min_ellipsoid"],
            free_n_conflict=sep_free["n_conflict"],
            res_min_ellipsoid=sep_res["min_ellipsoid"],
            res_n_conflict=sep_res["n_conflict"],
            res_min_horiz_nm=sep_res["min_horiz_nm"],
            res_vert_at_cpa_ft=sep_res["vert_at_cpa_ft"],
            vert_share=sep_res["vert_share"],
            eff_vert_ft=efforts["vert_ft"],
            eff_lat_km=efforts["lat_km"],
            eff_temp_s=efforts["temp_s"],
            mode=mode,
            **meta,
        )
        rows.append(row)
        print(
            f"  penalty {penalty_pct:5.2f}%  | free conflicts {sep_free['n_conflict']:2d}"
            f" -> resolved {sep_res['n_conflict']:2d} | min ellipsoid {sep_res['min_ellipsoid']:.3f}"
            f" | mode {mode}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
_MODE_COLORS = {
    "vertical": "#1f77b4",
    "lateral": "#d62728",
    "mixed": "#9467bd",
    "vertical+temporal": "#2ca02c",
    "lateral+temporal": "#ff7f0e",
    "mixed+temporal": "#8c564b",
    "none": "#7f7f7f",
}


def plot_penalty(df, cfg):
    ok = df[df.get("penalty_pct").notna()] if "penalty_pct" in df else df
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ok.angle_deg, ok.penalty_pct, "-", color="0.6", zorder=1)
    for mode, sub in ok.groupby("mode"):
        ax.scatter(
            sub.angle_deg,
            sub.penalty_pct,
            s=55,
            zorder=3,
            color=_MODE_COLORS.get(mode, "k"),
            label=mode,
            edgecolor="k",
            linewidth=0.5,
        )
    ax.set_xlabel("Crossing angle (deg)")
    ax.set_ylabel("Total fuel penalty vs unconstrained (%)")
    ax.set_title("Cost of separation vs encounter geometry")
    ax.grid(alpha=0.3)
    ax.legend(title="Resolution mode", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{cfg['out_prefix']}_penalty.png", dpi=160)
    plt.close(fig)


def plot_efforts(df, cfg):
    ok = df[df.get("penalty_pct").notna()] if "penalty_pct" in df else df
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(ok.angle_deg, ok.eff_vert_ft, "o-", label="vertical (ft)", color="#1f77b4")
    ax1b = ax1.twinx()
    ax1b.plot(ok.angle_deg, ok.eff_lat_km, "s--", label="lateral (km)", color="#d62728")
    ax1.plot(ok.angle_deg, ok.eff_temp_s, "^:", label="temporal (s)", color="#2ca02c")
    ax1.set_ylabel("vertical (ft) / temporal (s)")
    ax1b.set_ylabel("lateral path increase (km)")
    ax1.set_title("Resolution effort by mode")
    ax1.grid(alpha=0.3)
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)

    ax2.axhline(1.0, color="k", lw=1, ls="--", label="separation bound")
    ax2.plot(
        ok.angle_deg,
        ok.res_min_ellipsoid,
        "o-",
        color="#9467bd",
        label="achieved min ellipsoid",
    )
    ax2.plot(
        ok.angle_deg,
        ok.free_min_ellipsoid,
        "x:",
        color="0.5",
        label="unconstrained min ellipsoid",
    )
    ax2.set_xlabel("Crossing angle (deg)")
    ax2.set_ylabel("min ellipsoid value")
    ax2.set_title("Separation: induced conflict and its resolution")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{cfg['out_prefix']}_efforts.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg=CONFIG):
    df = run_sweep(cfg)
    csv_path = f"{cfg['out_prefix']}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}")

    if "penalty_pct" in df.columns and df.penalty_pct.notna().any():
        plot_penalty(df, cfg)
        plot_efforts(df, cfg)
        print(
            f"saved {cfg['out_prefix']}_penalty.png and {cfg['out_prefix']}_efforts.png"
        )

    print("\n=== summary ===")
    cols = [
        "angle_deg",
        "penalty_pct",
        "free_n_conflict",
        "res_n_conflict",
        "res_min_ellipsoid",
        "mode",
        "multi_iters",
        "multi_t_wall",
    ]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df[cols].to_string(index=False))
    return df


# %%

if __name__ == "__main__":
    main()

# %%
