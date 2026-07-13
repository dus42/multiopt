# %%
"""
Multi-aircraft arrival to a single runway
=========================================

3-4 aircraft start their descent from different distances, directions, and
departure times, and all must land on the SAME runway (same threshold, same
landing heading). Because they share the destination, separation cannot be
spatial at the threshold - they must be *sequenced* in time. This exercises the
separation constraint in the convergent-arrival regime, the complement to the
crossing/roundabout en-route tests.

Pipeline:
  1. conflict-free cruise solve -> supplies top-of-descent altitude (df_cruise)
  2. independent descents (conflict=False) -> baseline fuel + natural arrival
     times (shows whether/where they conflict if left unmanaged)
  3. joint descent (conflict=True) -> separated/sequenced arrivals
  4. post-hoc pairwise separation over each pair's both-airborne window,
     reporting horizontal AND vertical separation at every true (cylindrical)
     conflict
  5. fuel/penalty, arrival sequence + spacing, residual conflicts, figures

PREREQUISITES (fixes in descent.py; see review):
  * init_conditions: add `alt_start = kwargs.get("alt_start", None)`
  * ensure `import openap` (or replace openap.aero.* with oc.aero.*)
  * apply the conflict_guess fix (separate df_cruise from the descent
    warm-start / conflict-detection guess) so this harness can pass
    conflict_guess=df_s

How the guesses are routed:
  df_cruise (explicit) supplies the top-of-descent altitude for the descent
  init. conflict_guess=df_s (the independent descents) drives the
  encounter-focused candidate_sync_times AND warm-starts the joint descent
  states - mirroring the cruise two-stage workflow.

Outputs:
  arrival_results_summary.csv
  arrival_results_aircraft.csv
  arrival_topdown.png       (converging approaches, free vs sequenced)
  arrival_profiles.png      (altitude vs time + pairwise separation vs time)
  arrival_timeline.png      (arrival time at threshold per aircraft)
  arrival_map.png           (cartopy map, TransverseMercator centred on runway)
"""

import time
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openap
from openap import top
from traffic.data import airports

warnings.filterwarnings("ignore")

NM = openap.aero.nm
FT = openap.aero.ft
rwy = airports["EHAM"].runways.data.query("name=='18R'")
CONFIG = dict(
    runway=(rwy.latitude.iloc[0], rwy.longitude.iloc[0]),  # threshold (lat, lon)
    runway_dir=rwy.bearing.iloc[0],  # landing heading (deg) - approach from the north
    actype="A320",
    # Each aircraft: distance_km from threshold, bearing_deg of origin, tstart_s.
    # tstart is staggered so that closer aircraft depart later -> all converge
    # on the threshold at roughly the same time (~1350 s), which is what induces
    # the arrival conflict the constraint must resolve. Rule of thumb used:
    # tstart ~= target_arrival - dist_km*1000 / (~190 m/s descent ground speed).
    # CHECK `free_total_conflicts` in the output: if it is 0, no conflict was
    # induced -- nudge tstart so the independent arrival times coincide.
    arrivals=[
        dict(dist_km=100.0, bearing_deg=10.0, tstart=0, m0=0.8),
        dict(dist_km=100.0, bearing_deg=100.0, tstart=30, m0=0.72),
        dict(dist_km=100.0, bearing_deg=190.0, tstart=90, m0=0.9),
        dict(dist_km=100.0, bearing_deg=280.0, tstart=60, m0=0.65),
    ],
    alt_start_ft=30000.0,  # top-of-descent altitude
    max_nodes=30,
    max_iterations=15000,
    Rxy=5 * NM,
    Rz=1000 * FT,
    posthoc_dt=5.0,
    out_prefix="arrival",
    out_suffix="_same4",
)


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------
def make_arrival_scenarios(cfg):
    rwy_lat, rwy_lon = cfg["runway"]
    scenarios = []
    for i, a in enumerate(cfg["arrivals"]):
        o_lat, o_lon = openap.aero.latlon(
            rwy_lat, rwy_lon, a["dist_km"] * 1000.0, a["bearing_deg"]
        )
        scenarios.append(
            {
                "actype": cfg["actype"],
                "origin": (round(float(o_lat), 4), round(float(o_lon), 4)),
                "destination": (rwy_lat, rwy_lon),
                "m0": a["m0"],
                "tstart": a["tstart"],
                "id": i,
            }
        )
    return scenarios


def _proj(lon, lat, lat0, lon0):
    bearings = openap.aero.bearing(lat0, lon0, lat, lon) * np.pi / 180.0
    distances = openap.aero.distance(lat0, lon0, lat, lon)
    return distances * np.sin(bearings), distances * np.cos(bearings)


# ---------------------------------------------------------------------------
# Post-hoc pairwise separation (both-airborne window only)
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
        # no overlap in airborne time -> trivially separated (sequenced apart)
        return dict(
            min_horiz_nm=np.inf,
            min_vert_ft=np.inf,
            min_ellipsoid=np.inf,
            horiz_at_cpa_nm=np.inf,
            vert_at_cpa_ft=np.inf,
            is_conflict=False,
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

    horiz_nm = horiz / NM
    vert_ft = np.abs(dz) / FT
    # true cylindrical conflict: horizontal AND vertical both violated at once
    conflict_mask = (horiz < Rxy) & (np.abs(dz) < Rz)
    is_conflict = bool(conflict_mask.any())

    i = int(np.argmin(ell))  # binding point (closest approach in ellipsoid sense)
    return dict(
        min_horiz_nm=float(horiz.min() / NM),
        min_vert_ft=float(vert_ft.min()),
        min_ellipsoid=float(ell.min()),
        # separation at the binding point (min-ellipsoid instant):
        horiz_at_cpa_nm=float(horiz_nm[i]),
        vert_at_cpa_ft=float(vert_ft[i]),
        is_conflict=is_conflict,
        n_conflict=int(conflict_mask.sum()),  # cylindrical conflict-sample count
        t_cpa=float(tg[i]),
        tg=tg,
        horiz_nm=horiz_nm,
    )


def all_pairs(dfs, cfg):
    n = len(dfs)
    total = 0
    gmin_ell = np.inf
    gmin_h = np.inf
    series = {}
    conflicts = []  # detailed per-pair records where a conflict occurs
    for i, j in itertools.combinations(range(n), 2):
        s = pair_separation(dfs[i], dfs[j], cfg)
        total += s["n_conflict"]
        gmin_ell = min(gmin_ell, s["min_ellipsoid"])
        gmin_h = min(gmin_h, s["min_horiz_nm"])
        series[(i, j)] = (s["tg"], s["horiz_nm"])
        if s["is_conflict"]:
            conflicts.append(
                dict(
                    pair=(i, j),
                    t_cpa=s["t_cpa"],
                    horiz_nm=s["horiz_at_cpa_nm"],
                    vert_ft=s["vert_at_cpa_ft"],
                    min_horiz_nm=s["min_horiz_nm"],
                    min_vert_ft=s["min_vert_ft"],
                    min_ellipsoid=s["min_ellipsoid"],
                    n_samples=s["n_conflict"],
                )
            )
    return dict(
        total_conflicts=total,
        min_ellipsoid=gmin_ell,
        min_horiz_nm=gmin_h,
        series=series,
        conflicts=conflicts,
    )


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def fuel(df):
    return float(df.mass.iloc[0] - df.mass.iloc[-1])


def arrival_time(df):
    """Absolute time at the threshold (last node)."""
    return float(df.ts.iloc[-1])


def _path_len_km(df):
    d = openap.aero.distance(
        df.latitude.values[:-1],
        df.longitude.values[:-1],
        df.latitude.values[1:],
        df.longitude.values[1:],
    )
    return float(np.sum(d)) / 1000.0


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
    scenarios = make_arrival_scenarios(cfg)

    print("Cruise solve (conflict-free) for top-of-descent altitude...")
    cruise_opt = top.Cruise(
        scenarios=scenarios, conflict=False, max_nodes=10, debug=True
    )
    df_cruise = cruise_opt.trajectory(objective="fuel")

    common = dict(
        df_cruise=df_cruise, runway_dir=cfg["runway_dir"], alt_start=cfg["alt_start_ft"]
    )

    print("Independent descents (baseline)...")
    free_opt = top.Descent(
        scenarios=scenarios, conflict=False, max_nodes=cfg["max_nodes"]
    )
    t0 = time.perf_counter()
    df_s = free_opt.trajectory(objective="fuel", **common)
    t_free = time.perf_counter() - t0

    print("Joint descent with separation constraint (warm-started from baseline)...")
    joint_opt = top.Descent(
        scenarios=scenarios,
        conflict=True,
        max_nodes=cfg["max_nodes"],
        max_iterations=cfg["max_iterations"],
        debug=True,
    )
    t0 = time.perf_counter()
    # conflict_guess = the independent descents: drives encounter-focused
    # candidate_sync_times AND warm-starts the descent states (requires the
    # descent.py conflict_guess fix).
    df_m = joint_opt.trajectory(objective="fuel", conflict_guess=df_s, **common)
    t_joint = time.perf_counter() - t0

    meta = dict(
        t_free=t_free,
        t_joint=t_joint,
        **{f"joint_{k}": v for k, v in _stats(joint_opt).items()},
    )
    # dfs_m = pd.read_csv(f"data/exp6/arrivals_multi{cfg['out_suffix']}.csv").iloc[:, 1:]
    # df_m = []
    # for sc in dfs_m.scenario.unique():
    #     df_m.append(dfs_m.query("scenario==@sc"))
    # meta = dict(meta="no_meta")
    pd.concat(df_s).to_csv(
        f"data/exp6/arrivals_single{cfg['out_suffix']}.csv", index=False
    )
    pd.concat(df_m).to_csv(
        f"data/exp6/arrivals_multi{cfg['out_suffix']}.csv", index=False
    )
    return scenarios, df_s, df_m, meta


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_topdown(df_s, df_m, cfg):
    rwy_lat, rwy_lon = cfg["runway"]
    cmap = plt.cm.turbo(np.linspace(0, 1, len(df_m)))
    fig, (axf, axr) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, dfs, ttl in [
        (axf, df_s, "Independent descents"),
        (axr, df_m, "Sequenced (separation-constrained)"),
    ]:
        for i, df in enumerate(dfs):
            x, y = _proj(df.longitude.values, df.latitude.values, rwy_lat, rwy_lon)
            ax.plot(x / 1000, y / 1000, color=cmap[i], lw=2, label=f"AC{i}")
            ax.scatter(x[0] / 1000, y[0] / 1000, color=cmap[i], s=30, zorder=5)
        # runway threshold + landing direction arrow
        ax.scatter(0, 0, marker="*", c="k", s=140, zorder=6)
        ang = np.radians(90.0 - cfg["runway_dir"])  # heading -> math angle
        ax.annotate(
            "",
            xy=(12 * np.cos(ang), 12 * np.sin(ang)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
        ax.add_patch(plt.Circle((0, 0), cfg["Rxy"] / 1000, fill=False, ls=":", ec="k"))
        ax.set_aspect("equal")
        ax.set_title(ttl)
        ax.set_xlabel("east (km)")
        ax.grid(alpha=0.3)
    axf.set_ylabel("north (km)")
    axf.legend(fontsize=8)
    fig.suptitle("Arrivals to a single runway (top-down)")
    fig.tight_layout()
    fig.savefig(f"{cfg['out_prefix']}_topdown{cfg['out_suffix']}.png", dpi=160)
    plt.close(fig)


def plot_profiles(df_m, res, cfg):
    cmap = plt.cm.turbo(np.linspace(0, 1, len(df_m)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    for i, df in enumerate(df_m):
        ax1.plot(df.ts, df.altitude, color=cmap[i], label=f"AC{i}")
        ax1.scatter(df.ts.iloc[-1], df.altitude.iloc[-1], color=cmap[i], s=25, zorder=5)
    ax1.set_ylabel("altitude (ft)")
    ax1.set_title("Descent profiles (markers = touchdown)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, ncol=4)

    for (i, j), (tg, horiz) in res["series"].items():
        if len(tg):
            ax2.plot(tg, horiz, lw=0.9, alpha=0.7, label=f"{i}-{j}")
    ax2.axhline(cfg["Rxy"] / NM, color="r", ls="--", lw=1.2, label="5 nm")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("pairwise horizontal sep (nm)")
    ax2.set_title("Pairwise separation while both airborne")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(f"{cfg['out_prefix']}_profiles{cfg['out_suffix']}.png", dpi=160)
    plt.close(fig)


def plot_timeline(df_s, df_m, cfg):
    n = len(df_m)
    ts_free = [arrival_time(d) for d in df_s]
    ts_seq = [arrival_time(d) for d in df_m]
    cmap = plt.cm.turbo(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i in range(n):
        ax.scatter(ts_free[i], 1, color=cmap[i], s=70, marker="o")
        ax.scatter(ts_seq[i], 0, color=cmap[i], s=70, marker="s", label=f"AC{i}")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["sequenced", "independent"])
    ax.set_xlabel("arrival time at threshold (s)")
    ax.set_title("Runway arrival times: independent vs sequenced")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, ncol=n)
    fig.tight_layout()
    fig.savefig(f"{cfg['out_prefix']}_timeline{cfg['out_suffix']}.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cartopy map (adapted from the NEEDED departure/arrival plotting style)
# ---------------------------------------------------------------------------
def plot_map(df_s, df_m, cfg):
    """TransverseMercator map centred on the runway: independent (dashed) vs
    sequenced (solid) arrivals, one colour per aircraft. No waypoint/cluster
    labels, no population-cost contour."""
    import cartopy.crs as ccrs
    from cartopy.feature import BORDERS, COASTLINE
    import matplotlib.colors as mcolors

    rwy_lat, rwy_lon = cfg["runway"]
    # colors = list(mcolors.TABLEAU_COLORS.keys()) + ["b", "g", "y", "m", "c"]
    cmap = plt.cm.turbo(np.linspace(0, 1, len(df_m)))
    proj = ccrs.TransverseMercator(central_longitude=rwy_lon, central_latitude=rwy_lat)
    trans = ccrs.PlateCarree()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(projection=proj))
    ax.add_feature(BORDERS, linestyle="dotted", alpha=0.4)
    ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.4)

    all_lon = np.concatenate([d.longitude.values for d in (df_s + df_m)])
    all_lat = np.concatenate([d.latitude.values for d in (df_s + df_m)])
    ax.set_extent(
        [
            all_lon.min() - 0.3,
            all_lon.max() + 0.3,
            all_lat.min() - 0.3,
            all_lat.max() + 0.3,
        ]
    )

    for i in range(len(df_m)):
        # c = colors[i % len(colors)]
        ax.plot(
            df_s[i].longitude,
            df_s[i].latitude,
            color=cmap[i],
            lw=1.2,
            ls="dashed",
            transform=trans,
            label="Independent" if i == 0 else None,
        )
        ax.plot(
            df_m[i].longitude,
            df_m[i].latitude,
            color=cmap[i],
            lw=2.0,
            transform=trans,
            label="Sequenced" if i == 0 else None,
        )
        ax.scatter(
            df_m[i].longitude.iloc[0],
            df_m[i].latitude.iloc[0],
            color=cmap[i],
            s=25,
            transform=trans,
            zorder=5,
        )

    # runway drawn as a short segment along the landing heading
    rad = np.radians(cfg["runway_dir"])
    d = 0.08
    ax.plot(
        [rwy_lon - d * np.sin(rad), rwy_lon + d * np.sin(rad)],
        [rwy_lat - d * np.cos(rad), rwy_lat + d * np.cos(rad)],
        color="k",
        lw=3,
        transform=trans,
        solid_capstyle="butt",
        label="Runway",
    )

    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(
        f"{cfg['out_prefix']}_map{cfg['out_suffix']}.png", bbox_inches="tight", dpi=200
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Conflict report (horizontal + vertical at each true cylindrical conflict)
# ---------------------------------------------------------------------------
def report_conflicts(tag, r, cfg):
    print(f"\n=== CONFLICTS ({tag}) ===")
    if not r["conflicts"]:
        print("  none (all pairs separated)")
        return
    for c in r["conflicts"]:
        i, j = c["pair"]
        print(
            f"  AC{i}-AC{j} @ t={c['t_cpa']:.0f}s: "
            f"horiz {c['horiz_nm']:.2f} nm (bound {cfg['Rxy']/NM:.0f}), "
            f"vert {c['vert_ft']:.0f} ft (bound {cfg['Rz']/FT:.0f}) | "
            f"window min horiz {c['min_horiz_nm']:.2f} nm, "
            f"min vert {c['min_vert_ft']:.0f} ft, "
            f"ellipsoid {c['min_ellipsoid']:.3f}, {c['n_samples']} samples"
        )


def conflicts_dataframe(free, res):
    """Tidy per-conflict table across both solutions for saving to CSV."""
    rows = []
    for tag, r in [("independent", free), ("sequenced", res)]:
        for c in r["conflicts"]:
            rows.append(
                dict(
                    solution=tag,
                    ac_i=c["pair"][0],
                    ac_j=c["pair"][1],
                    t_cpa_s=c["t_cpa"],
                    horiz_at_cpa_nm=c["horiz_nm"],
                    vert_at_cpa_ft=c["vert_ft"],
                    min_horiz_nm=c["min_horiz_nm"],
                    min_vert_ft=c["min_vert_ft"],
                    min_ellipsoid=c["min_ellipsoid"],
                    n_conflict_samples=c["n_samples"],
                )
            )
    return pd.DataFrame(rows)


def conflict_instant(df1, df2, cfg):
    """Is the pair in conflict at any instant (horiz < Rxy AND vert < Rz)?
    If so, return horizontal and vertical distance at the most severe instant."""
    Rxy, Rz, dt = cfg["Rxy"], cfg["Rz"], cfg["posthoc_dt"]
    lat0 = 0.5 * (df1.latitude.mean() + df2.latitude.mean())
    lon0 = 0.5 * (df1.longitude.mean() + df2.longitude.mean())
    x1, y1 = _proj(df1.longitude.values, df1.latitude.values, lat0, lon0)
    x2, y2 = _proj(df2.longitude.values, df2.latitude.values, lat0, lon0)
    h1, h2 = df1.altitude.values * FT, df2.altitude.values * FT
    t1, t2 = df1.ts.values, df2.ts.values
    t0, tend = max(t1.min(), t2.min()), min(t1.max(), t2.max())
    if tend <= t0:
        return dict(conflict=False)

    tg = np.arange(t0, tend, dt)
    X1, Y1, H1 = (np.interp(tg, t1, a) for a in (x1, y1, h1))
    X2, Y2, H2 = (np.interp(tg, t2, a) for a in (x2, y2, h2))
    horiz = np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2)
    vert = np.abs(H1 - H2)

    mask = (horiz < Rxy) & (vert < Rz)  # conflict at each instant
    if not mask.any():
        return dict(conflict=False)

    ell = (horiz / Rxy) ** 2 + (vert / Rz) ** 8  # severity, restricted to conflicts
    k = int(np.argmin(np.where(mask, ell, np.inf)))
    return dict(
        conflict=True,
        t=float(tg[k]),
        horiz_nm=float(horiz[k] / NM),
        vert_ft=float(vert[k] / FT),
    )


# %%#---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg=CONFIG):
    scenarios, df_s, df_m, meta = run(cfg)

    free = all_pairs(df_s, cfg)
    res = all_pairs(df_m, cfg)

    rows = []
    for i in range(len(df_m)):
        rows.append(
            dict(
                aircraft=i,
                origin=scenarios[i]["origin"],
                tstart_s=scenarios[i]["tstart"],
                fuel_free_kg=fuel(df_s[i]),
                fuel_seq_kg=fuel(df_m[i]),
                penalty_kg=fuel(df_m[i]) - fuel(df_s[i]),
                arr_free_s=arrival_time(df_s[i]),
                arr_seq_s=arrival_time(df_m[i]),
                path_free_km=_path_len_km(df_s[i]),
                path_seq_km=_path_len_km(df_m[i]),
            )
        )
    ac_df = pd.DataFrame(rows)

    # arrival spacing in the sequenced solution
    seq_order = ac_df.sort_values("arr_seq_s")
    gaps = np.diff(seq_order.arr_seq_s.values)

    single_sum = sum(fuel(d) for d in df_s)
    multi_sum = sum(fuel(d) for d in df_m)
    summary = dict(
        n_aircraft=len(df_m),
        single_sum_kg=single_sum,
        multi_sum_kg=multi_sum,
        penalty_kg=multi_sum - single_sum,
        penalty_pct=100.0 * (multi_sum - single_sum) / single_sum,
        free_total_conflicts=free["total_conflicts"],
        seq_total_conflicts=res["total_conflicts"],
        free_conflicting_pairs=len(free["conflicts"]),
        seq_conflicting_pairs=len(res["conflicts"]),
        free_min_ellipsoid=free["min_ellipsoid"],
        seq_min_ellipsoid=res["min_ellipsoid"],
        free_min_horiz_nm=free["min_horiz_nm"],
        seq_min_horiz_nm=res["min_horiz_nm"],
        arrival_order=list(seq_order.aircraft.values),
        min_arrival_gap_s=float(gaps.min()) if len(gaps) else np.nan,
        **meta,
    )

    ac_df.to_csv(f"{cfg['out_prefix']}_results_aircraft.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        f"{cfg['out_prefix']}_results_summary{cfg['out_suffix']}.csv", index=False
    )

    # per-conflict table (horizontal + vertical) for both solutions
    conf_df = conflicts_dataframe(free, res)
    conf_df.to_csv(
        f"{cfg['out_prefix']}_results_conflicts{cfg['out_suffix']}.csv", index=False
    )

    plot_topdown(df_s, df_m, cfg)
    plot_profiles(df_m, res, cfg)
    plot_timeline(df_s, df_m, cfg)
    try:
        plot_map(df_s, df_m, cfg)
    except Exception as e:
        print(f"  (skipped cartopy map: {e})")

    # conflict report: horizontal + vertical separation at each true conflict
    report_conflicts("independent", free, cfg)
    report_conflicts("sequenced", res, cfg)

    print("\n=== ARRIVAL SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    print("\nper-aircraft:")
    with pd.option_context("display.width", 180):
        print(ac_df.to_string(index=False))

    print("\n=== CONFLICT CHECK ===")
    for i, j in itertools.combinations(range(len(df_m)), 2):
        c = conflict_instant(df_m[i], df_m[j], cfg)
        if c["conflict"]:
            print(
                f"  AC{i}-AC{j}: CONFLICT at t={c['t']:.0f}s -> "
                f"horiz {c['horiz_nm']:.2f} nm, vert {c['vert_ft']:.0f} ft"
            )
        else:
            print(f"  AC{i}-AC{j}: clear")
    return scenarios, df_s, df_m, summary, ac_df


# %%
if __name__ == "__main__":
    main()
# # %%

# dfs_m = pd.read_csv(f"data/exp6/arrivals_multi_same4.csv")
# df_m = []
# for sc in dfs_m.scenario.unique():
#     df_m.append(dfs_m.query("scenario==@sc"))
# dfs_s = pd.read_csv(f"data/exp6/arrivals_single_same4.csv")
# df_s = []
# for sc in dfs_s.scenario.unique():
#     df_s.append(dfs_s.query("scenario==@sc"))
# plot_topdown(df_s, df_m, CONFIG)
# %%
