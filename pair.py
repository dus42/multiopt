# %%
"""
Symmetric two-aircraft encounter experiment - opentop v2
========================================================

Two identical aircraft on straight tracks that cross at a common point at the
same time, swept over the angle between their headings. The geometry is
symmetric about due east, so both aircraft see the same encounter: same type,
same initial mass, same route length, same start time, mirrored bearings.
angle=180 is head-on on one great circle; angle=90 is orthogonal.

For each angle and each resolution mode:
  0. probe the fuel-optimal cruise altitude (level-flight solve, free level)
  1. free solve   - MultiAircraft(enforce_separation=False): the same NLP minus
     the separation terms, so the penalty is attributable to separation alone
  2. held solve   - separation enforced
  3. post-hoc verification on an independent dense absolute-time grid

RESOLUTION MODES
  both       lateral and vertical free; cruise descent enabled so the vertical
             channel works in both directions
  horizontal fix_cruise_altitude() + common_altitude=True: one shared optimised
             level, vertical term identically zero, lateral resolution only
  vertical   fix_track_angle(): constant heading, i.e. the straight route, which
             is the unconstrained lateral optimum in still air; altitude only

RECOVERY. Both endpoint altitudes are pinned to the probe optimum, so any
vertical manoeuvre is a temporary excursion rather than a permanent level
change (worst_alt_return_error_ft records the residual). End positions are hard
endpoint bounds in opentop, so every aircraft also finishes at its nominal
destination; worst_end_position_error_m logs that it really did.

IDENTICAL AIRCRAFT. Same type and mass means both want the same flight level,
so the encounter is genuine rather than resolved for free by stratification.
The cost is that the up/down assignment is a tie broken by opentop's
symmetry-breaking warm start, so in "both" mode the sign of individual
manoeuvres is warm-start dependent and the solver can wander in a flat region.
The constrained modes are the controlled experiments.

OUTPUTS (all under out_dir)
  cases.csv                 one row per (angle, mode)
  aircraft.csv              per aircraft: fuel, excursion, recovery, cross-track
  pairs.csv                 post-hoc analysis per pair, free and held
  solver_log.csv            status, iterations, constraints, timings per solve
  trajectories/<case>.csv   full state, free and held
  figures/<case>_tracks.png, _profiles.png, _separation.png
  figures/summary_penalty.png

  python pair_experiment.py --angles 30 90 150 180 --modes horizontal vertical
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openap
import opentop as top

warnings.filterwarnings("ignore")

NM, FT = openap.aero.nm, openap.aero.ft
# MODES = ("both", "horizontal", "vertical")
MODES = ("horizontal", "vertical")


# ---------------------------------------------------------------------------
# Cruise that departs from and returns to a given altitude
# ---------------------------------------------------------------------------
class ReturningCruise(top.Cruise):
    """Cruise pinned to `cruise_alt_m` at both ends, free in between.

    init_conditions() is where opentop builds the endpoint state boxes, and it
    runs inside _add_formulation - the one entry point shared by
    Cruise.trajectory() (the warm start) and MultiAircraft._build_problem (the
    joint solve). Tightening the boxes there applies in both, and they become
    plain IPOPT variable bounds rather than extra equality constraints.

    Needed because Cruise bounds the initial and final altitude independently
    inside [h_min, h_max], and allow_cruise_descent() removes the default
    vs >= 0. Together those let the solver descend the whole way and pay for
    range with potential energy instead of fuel, which is cheaper than the true
    cruise optimum because nothing charges for arriving low.
    """

    cruise_alt_m: float | None = None
    alt_tolerance_m: float = 1000.0

    def init_conditions(self, **kwargs):
        super().init_conditions(**kwargs)
        if self.cruise_alt_m is None:
            return
        low = self.cruise_alt_m - self.alt_tolerance_m
        high = self.cruise_alt_m + self.alt_tolerance_m
        for lower, upper in ((self.x_0_lb, self.x_0_ub), (self.x_f_lb, self.x_f_ub)):
            lower[2] = low
            upper[2] = high


def probe_optimum_m(cfg, origin, destination, cache):
    """Fuel-optimal cruise altitude from a level-flight solve.

    fix_cruise_altitude() with free endpoints leaves opentop one altitude to
    choose and it picks the cheapest. Agrees with a fixed-level fuel sweep
    minimum to within 0.05 kg. Cached per route length: a symmetric geometry
    costs one probe, not n.
    """
    key = round(float(openap.aero.distance(*origin, *destination)) / 1000.0, 1)
    if key in cache:
        return cache[key]
    probe = top.Cruise(cfg["actype"], origin, destination, m0=cfg["m0"])
    probe.setup(
        nodes=cfg["nodes"],
        max_iter=cfg["max_iterations"],
        tol=cfg["tol"],
        acceptable_tol=cfg["acceptable_tol"],
    )
    if cfg["exact_hessian"]:
        probe.solver_options["ipopt.hessian_approximation"] = "exact"
    probe.fix_cruise_altitude()
    df = probe.trajectory(objective=cfg["objective"], return_failed=True)
    if df is None or df.empty:
        raise RuntimeError("altitude probe failed")
    cache[key] = (
        float(df.altitude.iloc[0]) * FT,
        float(df.mass.iloc[0] - df.mass.iloc[-1]),
    )
    return cache[key]


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------
def make_optimizer(cfg, origin, destination, mode, h_opt_m, stage):
    """One Cruise configured for the resolution mode and the solve stage.

    mode="horizontal" -> fix_cruise_altitude(): dz == 0 for every pair, so only
        lateral manoeuvring can raise the metric. No altitude pinning needed.
    mode="vertical"   -> fix_track_angle(): constant heading, i.e. the straight
        route, which is the unconstrained lateral optimum in still air.
    mode="both"       -> nothing fixed.
    stage="free" additionally locks the track when straight_baseline is on:
        lateral position is a nearly flat direction of the fuel objective, so an
        unpinned floor wanders (~1.8 km of bow for 0.36 kg on an 816 kg flight),
        which is a sizeable fraction of a penalty of order 1 %.
    """
    pin = cfg["return_to_optimum"] and mode != "horizontal" and h_opt_m is not None
    factory = ReturningCruise if pin else top.Cruise
    opt = factory(cfg["actype"], origin, destination, m0=cfg["m0"])
    opt.setup(
        nodes=cfg["nodes"],
        max_iter=cfg["max_iterations"],
        tol=cfg["tol"],
        acceptable_tol=cfg["acceptable_tol"],
    )
    if cfg["exact_hessian"]:
        opt.solver_options["ipopt.hessian_approximation"] = "exact"
    if pin:
        opt.cruise_alt_m = h_opt_m
        opt.alt_tolerance_m = cfg["alt_tolerance_ft"] * FT

    if mode == "horizontal":
        opt.fix_cruise_altitude()
    elif cfg["allow_cruise_descent"]:
        # Vertical channel in play: allow descent too, otherwise resolution is
        # climb-only and an aircraft could never come back to the optimum.
        opt.allow_cruise_descent()

    if mode == "vertical" or (stage == "free" and cfg["straight_baseline"]):
        opt.fix_track_angle()
    return opt


def solve_fleet(cfg, legs, mode, h_opt_m, *, enforce, stage):
    flights = [
        top.FlightSpec(
            fid,
            make_optimizer(cfg, origin, destination, mode, h_opt_m, stage),
            start_time=start,
            objective=cfg["objective"],
        )
        for fid, origin, destination, start in legs
    ]
    t0 = time.perf_counter()
    result = top.MultiAircraft(
        flights,
        separation=top.SeparationConfig(
            horizontal_m=cfg["Rxy"],
            vertical_m=cfg["Rz"],
            vertical_power=cfg["vertical_power"],
            minimum_metric=cfg["minimum_metric"],
        ),
        enforce_separation=enforce,
        max_iter=cfg["max_iterations"],
        common_altitude=(mode == "horizontal"),
    ).trajectory()
    return result, time.perf_counter() - t0


def solver_log_row(case, stage, result, wall_s):
    """Everything worth knowing about the solve itself."""
    stats = dict(result.stats or {})
    return dict(
        case=case,
        stage=stage,
        status=str(result.status),
        solver_success=bool(result.solver_success),
        separation_success=bool(result.separation_success),
        objective=float(result.objective),
        nlp_variables=int(result.nlp_variables),
        nlp_constraints=int(result.nlp_constraints),
        separation_constraints=int(result.separation_constraints),
        refinement_rounds=int(result.refinement_rounds),
        iterations=int(stats.get("iter_count", -1)),
        build_time_s=float(result.build_time_s),
        solve_time_s=float(result.solve_time_s),
        verification_time_s=float(result.verification_time_s),
        wall_time_s=float(wall_s),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def fuel(df):
    return float(df.mass.iloc[0] - df.mass.iloc[-1])


def path_km(df):
    return (
        float(
            np.sum(
                openap.aero.distance(
                    df.latitude.values[:-1],
                    df.longitude.values[:-1],
                    df.latitude.values[1:],
                    df.longitude.values[1:],
                )
            )
        )
        / 1000.0
    )


def track_frame(df):
    """Along-track and signed cross-track (m) about the aircraft own chord."""
    x, y = df.x.to_numpy(float), df.y.to_numpy(float)
    vx, vy = x[-1] - x[0], y[-1] - y[0]
    length = np.hypot(vx, vy)
    if length == 0:
        return np.zeros_like(x), np.zeros_like(y)
    ux, uy = vx / length, vy / length
    dx, dy = x - x[0], y - y[0]
    return dx * ux + dy * uy, dx * uy - dy * ux


def cross_track_km(df):
    return float(np.abs(track_frame(df)[1]).max() / 1000.0)


def end_position_error_m(df, destination):
    """Distance from the flown end point to the nominal destination.

    The destination is a hard endpoint bound in opentop, so this should be ~0.
    Logged anyway: it is the cheap check that the geometry really closed.
    """
    return float(
        openap.aero.distance(
            float(df.latitude.iloc[-1]), float(df.longitude.iloc[-1]), *destination
        )
    )


def pair_metrics(df1, df2, cfg, prefix):
    """Dense-time pairwise check in the shared fleet projection (x, y, h).

    Independent of the solver own sampled constraint: opentop enforces the
    metric at selected times and verifies on a finer grid, so this is the
    outside check that the result actually holds.
    """
    t1 = df1.absolute_ts.to_numpy(float)
    t2 = df2.absolute_ts.to_numpy(float)
    lo, hi = max(t1.min(), t2.min()), min(t1.max(), t2.max())
    if hi <= lo:
        return {
            prefix + "min_metric": np.inf,
            prefix + "n_conflict": 0,
            prefix + "min_horiz_nm": np.inf,
            prefix + "vert_at_cpa_ft": np.inf,
            prefix + "max_vert_ft": 0.0,
            prefix + "t_cpa": np.nan,
            prefix + "vert_share": np.nan,
        }, None
    ts = np.unique(np.append(np.arange(lo, hi, cfg["posthoc_dt"]), hi))
    a = {c: np.interp(ts, t1, df1[c].to_numpy(float)) for c in "xyh"}
    b = {c: np.interp(ts, t2, df2[c].to_numpy(float)) for c in "xyh"}
    dx, dy, dz = a["x"] - b["x"], a["y"] - b["y"], a["h"] - b["h"]
    horizontal = (dx**2 + dy**2) / cfg["Rxy"] ** 2
    vertical = (dz / cfg["Rz"]) ** cfg["vertical_power"]
    m = horizontal + vertical
    i = int(np.argmin(m))
    series = pd.DataFrame(
        dict(ts=ts, horiz_nm=np.hypot(dx, dy) / NM, vert_ft=np.abs(dz) / FT, metric=m)
    )
    return {
        prefix + "min_metric": float(m[i]),
        prefix + "n_conflict": int((m < 1.0).sum()),  # inside the protected volume
        prefix + "min_horiz_nm": float(np.hypot(dx, dy).min() / NM),
        prefix + "vert_at_cpa_ft": float(abs(dz[i]) / FT),
        prefix + "hor_at_cpa_ft": float(np.sqrt(dx[i] ** 2 + dy[i] ** 2) / NM),
        prefix + "max_vert_ft": float(np.abs(dz).max() / FT),
        prefix + "t_cpa": float(ts[i]),
        prefix + "vert_share": float(vertical[i] / (m[i] + 1e-12)),
    }, series


def classify(rows, max_temporal_s):
    """Heuristic label for the dominant channel at the binding pair. Report it
    as a heuristic, not a taxonomy."""
    finite = [r for r in rows if np.isfinite(r["held_min_metric"])]
    if not finite:
        return "none"
    share = min(finite, key=lambda r: r["held_min_metric"])["held_vert_share"]
    if np.isnan(share):
        return "none"
    label = "vertical" if share > 0.6 else "lateral" if share < 0.4 else "mixed"
    return label + "+temporal" if max_temporal_s > 30.0 else label


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_tracks(cfg, case, df_free, df_held, ids, out):
    """Full routes, a zoom at the scale of the protected volume, and signed
    cross-track. The zoom and cross-track panels exist because a resolution
    offset of a few km is invisible on a +-300 km axis."""
    colors = plt.cm.turbo(np.linspace(0, 1, len(ids)))
    x0 = np.mean([d.x.mean() for d in df_free])
    y0 = np.mean([d.y.mean() for d in df_free])
    zoom = max(3.0 * cfg["Rxy"] / 1000.0, 100.0)

    fig, (ax_full, ax_zoom, ax_cross) = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, limit, title in [
        (ax_full, None, "Full routes"),
        (ax_zoom, zoom, f"Conflict point, +-{zoom:.0f} km"),
    ]:
        for k, (a, b) in enumerate(zip(df_free, df_held)):
            for d, style, alpha in ((a, "--", 0.45), (b, "-", 1.0)):
                ax.plot(
                    (d.x.to_numpy(float) - x0) / 1000,
                    (d.y.to_numpy(float) - y0) / 1000,
                    style,
                    color=colors[k],
                    lw=1.8,
                    alpha=alpha,
                    label=ids[k] if style == "-" else None,
                )
        theta = np.linspace(0, 2 * np.pi, 200)
        r = cfg["Rxy"] / 1000
        ax.plot(r * np.cos(theta), r * np.sin(theta), "k:", lw=1.2)
        ax.scatter(0, 0, marker="x", c="k", s=70, zorder=6)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("east (km)")
        ax.grid(alpha=0.3)
        if limit:
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
    ax_full.set_ylabel("north (km)")
    ax_full.legend(fontsize=8, title="solid=held\ndashed=free")

    for k, (a, b) in enumerate(zip(df_free, df_held)):
        for d, style, alpha in ((a, "--", 0.45), (b, "-", 1.0)):
            along, cross = track_frame(d)
            ax_cross.plot(
                along / 1000, cross / 1000, style, color=colors[k], lw=1.8, alpha=alpha
            )
    ax_cross.axhline(0, color="k", lw=0.8)
    ax_cross.set_title("Cross-track deviation from own straight route")
    ax_cross.set_xlabel("along-track (km)")
    ax_cross.set_ylabel("cross-track (km)")
    ax_cross.grid(alpha=0.3)

    fig.suptitle(case)
    fig.tight_layout()
    fig.savefig(f"{out}/figures/{case}_tracks.png", dpi=160)
    plt.close(fig)


def fig_profiles(cfg, case, df_free, df_held, ids, h_opt_ft, out):
    colors = plt.cm.turbo(np.linspace(0, 1, len(ids)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for k, (a, b) in enumerate(zip(df_free, df_held)):
        ax1.plot(a.absolute_ts, a.altitude, "--", color=colors[k], alpha=0.45)
        ax1.plot(b.absolute_ts, b.altitude, "-", color=colors[k], lw=1.8, label=ids[k])
        ax2.plot(b.absolute_ts, b.mach, "-", color=colors[k], lw=1.5)
    if np.isfinite(h_opt_ft):
        ax1.axhline(
            h_opt_ft, color="k", ls="--", lw=1.0, label=f"optimum {h_opt_ft:.0f} ft"
        )
    ax1.set_ylabel("altitude (ft)")
    ax1.set_title(f"{case} - altitude (dashed = free, solid = held)")
    ax1.legend(fontsize=8, ncol=3)
    ax2.set_ylabel("Mach")
    ax2.set_xlabel("absolute time (s)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/figures/{case}_profiles.png", dpi=160)
    plt.close(fig)


def fig_separation(cfg, case, series_free, series_held, out):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for label, series, style, alpha in (
        ("free", series_free, "--", 0.5),
        ("held", series_held, "-", 1.0),
    ):
        for s in series:
            if s is None:
                continue
            ax1.plot(
                s.ts,
                s.horiz_nm,
                style,
                lw=1.0,
                alpha=alpha,
                color="C0" if label == "held" else "0.6",
            )
            ax2.plot(
                s.ts,
                s.vert_ft,
                style,
                lw=1.0,
                alpha=alpha,
                color="C1" if label == "held" else "0.6",
            )
            ax3.plot(
                s.ts,
                s.metric,
                style,
                lw=1.0,
                alpha=alpha,
                color="C2" if label == "held" else "0.6",
            )
    ax1.axhline(cfg["Rxy"] / NM, color="r", ls="--", lw=1.2, label="5 nm")
    ax1.set_ylabel("horizontal (nm)")
    ax2.axhline(cfg["Rz"] / FT, color="r", ls="--", lw=1.2, label="1000 ft")
    ax2.set_ylabel("vertical (ft)")
    ax3.axhline(cfg["minimum_metric"], color="k", ls="--", lw=1.2, label="bound 1.3")
    ax3.axhline(1.0, color="r", ls=":", lw=1.2, label="protected volume")
    ax3.set_yscale("log")
    ax3.set_ylabel("separation metric")
    ax3.set_xlabel("absolute time (s)")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{case} - pairwise separation (grey = free, colour = held)")
    fig.tight_layout()
    fig.savefig(f"{out}/figures/{case}_separation.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------
def run_case(cfg, case, alt_cache):
    legs = case["legs"]
    ids = [fid for fid, *_ in legs]
    name = case["name"]
    out = cfg["out_dir"]

    h_opt_m, probe_fuel = np.nan, np.nan
    if cfg["return_to_optimum"]:
        h_opt_m, probe_fuel = probe_optimum_m(cfg, legs[0][1], legs[0][2], alt_cache)

    free, wall_free = solve_fleet(
        cfg, legs, case["mode"], h_opt_m, enforce=False, stage="free"
    )
    held, wall_held = solve_fleet(
        cfg, legs, case["mode"], h_opt_m, enforce=True, stage="held"
    )
    df_free = [free.trajectories[fid] for fid in ids]
    df_held = [held.trajectories[fid] for fid in ids]

    reference_ft = np.nan if not np.isfinite(h_opt_m) else h_opt_m / FT

    aircraft = []
    for k, fid in enumerate(ids):
        base = (
            reference_ft if np.isfinite(reference_ft) else df_held[k].altitude.iloc[0]
        )
        aircraft.append(
            dict(
                case=name,
                aircraft=fid,
                actype=cfg["actype"],
                m0=cfg["m0"],
                fuel_free_kg=fuel(df_free[k]),
                fuel_held_kg=fuel(df_held[k]),
                penalty_kg=fuel(df_held[k]) - fuel(df_free[k]),
                optimum_alt_ft=reference_ft,
                alt_start_ft=float(df_held[k].altitude.iloc[0]),
                alt_end_ft=float(df_held[k].altitude.iloc[-1]),
                alt_excursion_ft=float(
                    np.max(np.abs(df_held[k].altitude.to_numpy(float) - base))
                ),
                alt_return_error_ft=float(abs(df_held[k].altitude.iloc[-1] - base)),
                cross_track_km=cross_track_km(df_held[k]),
                baseline_cross_track_km=cross_track_km(df_free[k]),
                path_increase_km=path_km(df_held[k]) - path_km(df_free[k]),
                duration_change_s=float(df_held[k].ts.iloc[-1])
                - float(df_free[k].ts.iloc[-1]),
                duration_s=float(df_held[k].ts.iloc[-1]),
                end_position_error_m=end_position_error_m(df_held[k], legs[k][2]),
            )
        )
    ac_df = pd.DataFrame(aircraft)

    pair_rows, series_free, series_held = [], [], []
    for i, j in itertools.combinations(range(len(ids)), 2):
        free_stats, free_series = pair_metrics(df_free[i], df_free[j], cfg, "free_")
        held_stats, held_series = pair_metrics(df_held[i], df_held[j], cfg, "held_")
        pair_rows.append(
            dict(case=name, first=ids[i], second=ids[j], **free_stats, **held_stats)
        )
        series_free.append(free_series)
        series_held.append(held_series)
    pair_df = pd.DataFrame(pair_rows)

    solver_pairs = pd.DataFrame([asdict(r) for r in held.pair_reports])
    free_sum, held_sum = ac_df.fuel_free_kg.sum(), ac_df.fuel_held_kg.sum()
    max_temporal = float(ac_df.duration_change_s.abs().max())

    case_row = dict(
        case=name,
        family=case["family"],
        mode=case["mode"],
        **case["params"],
        n_aircraft=len(ids),
        actype=cfg["actype"],
        m0=cfg["m0"],
        nodes=cfg["nodes"],
        route_km=cfg["route_km"],
        optimum_alt_ft=reference_ft,
        level_probe_fuel_kg=probe_fuel,
        free_fuel_kg=free_sum,
        held_fuel_kg=held_sum,
        penalty_kg=held_sum - free_sum,
        penalty_pct=100.0 * (held_sum - free_sum) / free_sum if free_sum else np.nan,
        free_n_conflict=int(pair_df.free_n_conflict.sum()),
        held_n_conflict=int(pair_df.held_n_conflict.sum()),
        free_min_metric=float(pair_df.free_min_metric.min()),
        held_min_metric=float(pair_df.held_min_metric.min()),
        held_min_horiz_nm=float(pair_df.held_min_horiz_nm.min()),
        held_max_vert_ft=float(pair_df.held_max_vert_ft.max()),
        solver_min_metric=(
            float(solver_pairs.minimum_metric.min()) if len(solver_pairs) else np.nan
        ),
        resolution=classify(pair_rows, max_temporal),
        max_alt_excursion_ft=float(ac_df.alt_excursion_ft.max()),
        worst_alt_return_error_ft=float(ac_df.alt_return_error_ft.max()),
        worst_end_position_error_m=float(ac_df.end_position_error_m.max()),
        max_cross_track_km=float(ac_df.cross_track_km.max()),
        max_baseline_cross_track_km=float(ac_df.baseline_cross_track_km.max()),
        max_path_increase_km=float(ac_df.path_increase_km.max()),
        max_temporal_change_s=max_temporal,
        free_success=bool(free.solver_success),
        held_success=bool(held.success),
        separation_success=bool(held.separation_success),
        status=str(held.status),
        refinement_rounds=int(held.refinement_rounds),
        separation_constraints=int(held.separation_constraints),
        nlp_variables=int(held.nlp_variables),
        solve_time_s=float(held.solve_time_s),
        wall_free_s=wall_free,
        wall_held_s=wall_held,
    )

    log_df = pd.DataFrame(
        [
            solver_log_row(name, "free", free, wall_free),
            solver_log_row(name, "held", held, wall_held),
        ]
    )

    trajectories = pd.concat(
        [
            pd.concat(
                [d.assign(aircraft=ids[k]) for k, d in enumerate(df_free)]
            ).assign(stage="free"),
            pd.concat(
                [d.assign(aircraft=ids[k]) for k, d in enumerate(df_held)]
            ).assign(stage="held"),
        ]
    ).assign(case=name)
    trajectories.to_csv(f"{out}/trajectories/{name}.csv", index=False)

    fig_tracks(cfg, name, df_free, df_held, ids, out)
    fig_profiles(cfg, name, df_free, df_held, ids, reference_ft, out)
    fig_separation(cfg, name, series_free, series_held, out)

    return case_row, ac_df, pair_df, log_df


CONFIG = dict(
    # ---- controlled ---------------------------------------------------------
    actype="A320",
    m0=0.80,
    route_km=600.0,
    center=(51.0, 7.0),
    nodes=30,
    max_iterations=3000,
    objective="fuel",
    tol=1e-8,
    acceptable_tol=1e-6,
    exact_hessian=True,
    # ---- altitude boundary condition ---------------------------------------
    return_to_optimum=True,
    alt_tolerance_ft=0.0,
    allow_cruise_descent=True,
    straight_baseline=True,
    # ---- separation model ---------------------------------------------------
    Rxy=5 * NM,
    Rz=1000 * FT,
    vertical_power=8,
    minimum_metric=1.3,
    posthoc_dt=2.0,
    # ---- sweep --------------------------------------------------------------
    # angles_deg=(30, 60, 90, 120, 150, 180),
    angles_deg=np.arange(30, 181, 30).tolist(),
    modes=("both", "horizontal", "vertical"),
    out_dir="results_pair",
)

HEADER = (
    "{n} case(s) | {actype} at m0={m0} | {route_km:.0f} km symmetric tracks "
    "| nodes={nodes} | modes={modes} | angles={angles_deg}"
)
REPORT_COLS = [
    "case",
    "angle_deg",
    "mode",
    "penalty_kg",
    "penalty_pct",
    "free_n_conflict",
    "held_n_conflict",
    "held_min_metric",
    "held_min_horiz_nm",
    "held_max_vert_ft",
    "resolution",
    "max_alt_excursion_ft",
    "worst_alt_return_error_ft",
    "worst_end_position_error_m",
    "max_cross_track_km",
    "refinement_rounds",
    "separation_constraints",
    "solve_time_s",
    "status",
]
PIVOT_INDEX = "angle_deg"


def pair_geometry(cfg, angle_deg):
    """Two tracks crossing at the centre with `angle_deg` between headings,
    mirrored about due east so the geometry is symmetric."""
    lat_c, lon_c = cfg["center"]
    half = cfg["route_km"] * 1000.0 / 2.0
    legs = []
    for k, bearing in enumerate([90.0 - angle_deg / 2.0, 90.0 + angle_deg / 2.0]):
        o = openap.aero.latlon(lat_c, lon_c, half, (bearing + 180.0) % 360.0)
        d = openap.aero.latlon(lat_c, lon_c, half, bearing)
        legs.append(
            (f"AC{k}", (float(o[0]), float(o[1])), (float(d[0]), float(d[1])), 0.0)
        )
    return legs


def build_cases(cfg):
    return [
        dict(
            name=f"pair_a{angle:03.0f}_{mode}",
            family="pair",
            mode=mode,
            legs=pair_geometry(cfg, angle),
            params=dict(angle_deg=float(angle)),
        )
        for mode in cfg["modes"]
        for angle in cfg["angles_deg"]
    ]


def summary_figure(df, cfg):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for mode, sub in df.groupby("mode"):
        sub = sub.sort_values("angle_deg")
        ax1.plot(sub.angle_deg, sub.penalty_pct, "o-", label=mode)
        ax2.plot(sub.angle_deg, sub.held_min_metric, "o-", label=mode)
    ax1.set_ylabel("fuel penalty vs free (%)")
    ax1.set_title(
        f"Cost of separation vs crossing angle ({cfg['actype']}, m0={cfg['m0']})"
    )
    ax2.axhline(cfg["minimum_metric"], color="k", ls="--", lw=1, label="bound 1.3")
    ax2.axhline(1.0, color="r", ls=":", lw=1, label="protected volume")
    ax2.set_yscale("log")
    ax2.set_ylabel("worst held metric")
    ax2.set_xlabel("crossing angle (deg)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{cfg['out_dir']}/figures/summary_penalty.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------
def main(cfg=None, **overrides):
    cfg = dict(CONFIG if cfg is None else cfg)
    cfg.update(overrides)
    out = cfg["out_dir"]
    os.makedirs(f"{out}/trajectories", exist_ok=True)
    os.makedirs(f"{out}/figures", exist_ok=True)

    cases = build_cases(cfg)
    print(HEADER.format(n=len(cases), **cfg))

    case_rows, ac_frames, pair_frames, log_frames = [], [], [], []
    alt_cache = {}
    for index, case in enumerate(cases, 1):
        print(f"\n[{index}/{len(cases)}] {case['name']}")
        try:
            case_row, ac_df, pair_df, log_df = run_case(cfg, case, alt_cache)
        except Exception as exc:  # keep the suite going if one case fails
            print(f"    FAILED: {exc}")
            case_rows.append(
                dict(
                    case=case["name"],
                    family=case["family"],
                    mode=case["mode"],
                    **case["params"],
                    error=str(exc),
                )
            )
            continue

        case_rows.append(case_row)
        ac_frames.append(ac_df)
        pair_frames.append(pair_df)
        log_frames.append(log_df)
        print(
            f"    penalty {case_row['penalty_kg']:8.2f} kg"
            f" ({case_row['penalty_pct']:6.3f} %)"
            f" | conflicts {case_row['free_n_conflict']:4d} -> {case_row['held_n_conflict']:3d}"
            f" | min metric {case_row['held_min_metric']:8.3f}"
            f" | {case_row['resolution']:<17}"
            f" | {case_row['status']}"
            f" | {case_row['separation_constraints']:3d} sep constraints"
            f" | {case_row['solve_time_s']:6.1f} s"
        )
        # incremental save: a long suite should not lose everything on a crash
        pd.DataFrame(case_rows).to_csv(f"{out}/cases.csv", index=False)
        pd.concat(ac_frames).to_csv(f"{out}/aircraft.csv", index=False)
        pd.concat(pair_frames).to_csv(f"{out}/pairs.csv", index=False)
        pd.concat(log_frames).to_csv(f"{out}/solver_log.csv", index=False)

    cases_df = pd.DataFrame(case_rows)
    if "penalty_pct" in cases_df and cases_df.penalty_pct.notna().any():
        summary_figure(cases_df[cases_df.penalty_pct.notna()], cfg)

    print(
        f"\nwrote {out}/cases.csv, aircraft.csv, pairs.csv, solver_log.csv,"
        f" trajectories/*.csv, figures/*.png"
    )
    cols = [c for c in REPORT_COLS if c in cases_df.columns]
    print("\n=== CASES ===")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(cases_df[cols].round(3).to_string(index=False))
    if "penalty_pct" in cases_df and cases_df.penalty_pct.notna().any():
        print("\n=== penalty (%) ===")
        print(
            cases_df.pivot_table(
                index=PIVOT_INDEX, columns="mode", values="penalty_pct"
            )
            .round(3)
            .to_string()
        )
        print("\n=== worst held metric (bound 1.3) ===")
        print(
            cases_df.pivot_table(
                index=PIVOT_INDEX, columns="mode", values="held_min_metric"
            )
            .round(3)
            .to_string()
        )
    return cases_df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="symmetric two-aircraft encounter sweep")
    p.add_argument("--angles", type=float, nargs="+", default=None)
    p.add_argument("--modes", nargs="+", choices=MODES, default=None)
    p.add_argument("--nodes", type=int, default=CONFIG["nodes"])
    p.add_argument("--route-km", type=float, default=CONFIG["route_km"])
    p.add_argument("--actype", default=CONFIG["actype"])
    p.add_argument("--m0", type=float, default=CONFIG["m0"])
    p.add_argument(
        "--no-recovery",
        action="store_true",
        help="do not pin the endpoint altitudes to the fuel optimum",
    )
    p.add_argument("--out-dir", default=CONFIG["out_dir"])
    a = p.parse_args(argv)
    overrides = dict(
        nodes=a.nodes,
        route_km=a.route_km,
        actype=a.actype,
        m0=a.m0,
        return_to_optimum=not a.no_recovery,
        out_dir=a.out_dir,
    )
    if a.angles:
        overrides["angles_deg"] = tuple(a.angles)
    if a.modes:
        overrides["modes"] = tuple(a.modes)
    return overrides


# %%
if __name__ == "__main__":
    main(**parse_args())
# %%
