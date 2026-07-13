# %% Head on Conflict
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openap import top
import openap

warnings.filterwarnings("ignore")

# %%
Rxy = 5 * openap.aero.nm
Rz = 1000 * openap.aero.ft

scenarios = [
    {
        "actype": "B738",
        "origin": (52.3, 9.8),
        "destination": (49.5, 2.5),
        "m0": 0.8,
        "tstart": 0,
        "id": 0,
    },
    {
        "actype": "B737",
        "origin": (49.5, 2.5),
        "destination": (52.3, 9.8),
        "m0": 0.85,
        "tstart": 0,
        "id": 1,
    },
]

# %%
# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import LineCollection


def plot_trajectories_map(
    dfs, scenarios=None, title="Optimized trajectories", color_by="altitude"
):
    """Plot trajectories on a cartopy map.

    dfs        : list of trajectory DataFrames (need longitude, latitude, altitude)
    scenarios  : optional list of scenario dicts (used for actype labels)
    color_by   : "altitude" -> colour each track by altitude; else flat colours
    """
    all_lon = np.concatenate([d.longitude.values for d in dfs])
    all_lat = np.concatenate([d.latitude.values for d in dfs])

    margin = 1.0
    extent = [
        all_lon.min() - margin,
        all_lon.max() + margin,
        all_lat.min() - margin,
        all_lat.max() + margin,
    ]

    proj = ccrs.LambertConformal(
        central_longitude=(all_lon.min() + all_lon.max()) / 2,
        central_latitude=(all_lat.min() + all_lat.max()) / 2,
    )
    transform = ccrs.PlateCarree()

    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=transform)

    ax.add_feature(cfeature.LAND, facecolor="#f3f3f1")
    ax.add_feature(cfeature.OCEAN, facecolor="#dceaf2")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    alt_min = min(d.altitude.min() for d in dfs)
    alt_max = max(d.altitude.max() for d in dfs)

    line = None
    for i, df in enumerate(dfs):
        lon = df.longitude.values
        lat = df.latitude.values
        label = scenarios[i]["actype"] if scenarios else f"AC{i}"

        if color_by == "altitude":
            pts = np.array([lon, lat]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(
                segs,
                cmap="viridis",
                norm=plt.Normalize(alt_min, alt_max),
                transform=transform,
                linewidth=2.5,
                zorder=5,
            )
            lc.set_array(df.altitude.values[:-1])
            line = ax.add_collection(lc)
        else:
            ax.plot(lon, lat, transform=transform, lw=2.5, zorder=5, label=label)

        # departure (circle) and arrival (square)
        ax.scatter(
            lon[0], lat[0], marker="o", c="k", s=40, transform=transform, zorder=6
        )
        ax.scatter(
            lon[-1], lat[-1], marker="s", c="k", s=40, transform=transform, zorder=6
        )
        ax.text(lon[0], lat[0], f"  {label}", transform=transform, fontsize=9, zorder=7)

    if color_by == "altitude" and line is not None:
        cbar = fig.colorbar(line, ax=ax, shrink=0.7, pad=0.06)
        cbar.set_label("Altitude (ft)")
    else:
        ax.legend(loc="best")

    plt.title(title)
    plt.tight_layout()
    plt.show()


def proj(lon, lat, lat0, lon0, inverse=False):
    """Project coordinates to/from local tangent plane."""
    if not inverse:
        bearings = openap.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159
        distances = openap.aero.distance(lat0, lon0, lat, lon)
        x = distances * np.sin(bearings)
        y = distances * np.cos(bearings)
        return x, y
    else:
        x, y = lon, lat
        distances = np.sqrt(x**2 + y**2)
        bearing = np.arctan2(x, y) * 180 / 3.14159
        lat, lon = openap.aero.latlon(lat0, lon0, distances, bearing)
        return lon, lat


def interpolate_trajectory(df, t_global, time_col="ts"):
    """Interpolate trajectory to common time grid."""
    df = df.sort_values(time_col)

    # Keep only times inside the actual trajectory span
    t_min, t_max = df[time_col].min(), df[time_col].max()
    t_use = t_global[(t_global >= t_min) & (t_global <= t_max)]

    if len(t_use) == 0:
        return pd.DataFrame(columns=df.columns)

    df_new = pd.DataFrame({time_col: t_use})

    # Columns to interpolate
    numeric_cols = df.select_dtypes(include="number").columns.drop(time_col)

    for col in numeric_cols:
        df_new[col] = np.interp(t_use, df[time_col], df[col])

    # Copy non-numeric columns
    for col in df.columns.difference(numeric_cols.tolist() + [time_col]):
        df_new[col] = df[col].iloc[0]

    return df_new


def analyze_conflict(df1, df2, Rxy, Rz, title=""):
    """
    Analyze and visualize conflicts between two trajectories.

    Returns:
        dfc: DataFrame with merged trajectory data and conflict detection
        mask: Boolean mask of conflict points
    """
    # Compute projection center
    lon0 = (
        (df1.longitude.iloc[0] + df1.longitude.iloc[-1]) / 2
        + (df1.longitude.iloc[0] + df1.longitude.iloc[-1]) / 2
    ) / 2
    lat0 = (
        (df1.latitude.iloc[0] + df1.latitude.iloc[-1]) / 2
        + (df1.latitude.iloc[0] + df1.latitude.iloc[-1]) / 2
    ) / 2

    # Project trajectories
    x, y = proj(df1.longitude.values, df1.latitude.values, lat0, lon0)
    df1.loc[:, "x"] = x
    df1.loc[:, "y"] = y
    x, y = proj(df2.longitude.values, df2.latitude.values, lat0, lon0)
    df2.loc[:, "x"] = x
    df2.loc[:, "y"] = y

    # Interpolate to common time grid
    t_start_global = min(df1.ts.iloc[0], df2.ts.iloc[0])
    t_end_global = int(max(df1.ts.iloc[-1], df2.ts.iloc[-1]))
    t_global = np.arange(t_start_global, t_end_global, 50)

    df1n = interpolate_trajectory(df1, t_global)
    df2n = interpolate_trajectory(df2, t_global)

    # Cut to common length
    df1c = df1n.iloc[: min(len(df1n), len(df2n)), :]
    df2c = df2n.iloc[: min(len(df1n), len(df2n)), :]

    # Merge trajectories
    dfc = df1c[["ts", "x", "y", "h"]].merge(df2c[["ts", "x", "y", "h"]], on="ts")

    # Detect conflicts
    dfc = dfc.assign(
        dist_x=dfc.x_x - dfc.x_y,
        dist_y=dfc.y_x - dfc.y_y,
        dist_z=dfc.h_x - dfc.h_y,
    ).assign(dist_lat=lambda x: (x.dist_x**2 + x.dist_y**2) ** 0.5)

    idx = dfc.query("(dist_lat < @Rxy) and (dist_z.abs() < @Rz)").index
    mask = idx
    if len(dfc.query("(dist_lat < @Rxy) and (dist_z.abs() < @Rz)")) > 0:
        print(
            dfc.query("(dist_lat < @Rxy) and (dist_z.abs() < @Rz)")[
                ["ts", "dist_lat", "dist_x", "dist_y", "dist_z"]
            ]
        )
        print(
            dfc.query("(dist_lat < @Rxy) and (dist_z.abs() < @Rz)").dist_lat.values
            / openap.aero.nm,
            dfc.query("(dist_lat < @Rxy) and (dist_z.abs() < @Rz)").dist_z.values
            / openap.aero.ft,
        )
    else:
        min_lat = dfc.query("dist_lat==dist_lat.min()").index[0]
        min_z = dfc.query("dist_z==dist_z.min()").index[0]
        print(
            dfc.loc[[min_lat, min_z], ["ts", "dist_lat", "dist_x", "dist_y", "dist_z"]]
        )
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(dfc.ts, dfc.x_x)
    ax1.plot(dfc.ts, dfc.x_y)
    ax2.plot(dfc.ts, dfc.y_x)
    ax2.plot(dfc.ts, dfc.y_y)
    ax3.plot(dfc.ts, dfc.h_x / openap.aero.ft)
    ax3.plot(dfc.ts, dfc.h_y / openap.aero.ft)

    ax1.scatter(dfc.ts, dfc.x_x, s=3)
    ax1.scatter(dfc.ts, dfc.x_y, s=3)
    ax2.scatter(dfc.ts, dfc.y_x, s=3)
    ax2.scatter(dfc.ts, dfc.y_y, s=3)
    ax3.scatter(dfc.ts, dfc.h_x / openap.aero.ft, s=3)
    ax3.scatter(dfc.ts, dfc.h_y / openap.aero.ft, s=3)

    ax1.scatter(dfc.ts.values[mask], dfc.x_x.values[mask], c="r", s=10)
    ax1.scatter(dfc.ts.values[mask], dfc.x_y.values[mask], c="r", s=10)
    ax2.scatter(dfc.ts.values[mask], dfc.y_x.values[mask], c="r", s=10)
    ax2.scatter(dfc.ts.values[mask], dfc.y_y.values[mask], c="r", s=10)
    ax3.scatter(dfc.ts.values[mask], dfc.h_x.values[mask] / openap.aero.ft, c="r", s=10)
    ax3.scatter(
        dfc.ts.values[mask],
        dfc.h_y.values[mask] / openap.aero.ft,
        c="r",
        s=10,
        label="conflict",
    )

    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax3.set_ylabel("altitude")
    ax3.set_xlabel("t")
    plt.legend()
    plt.suptitle(title)
    plt.show()

    return dfc, mask


def compute_ellipsoid(dfc, Rxy, Rz):
    """Compute ellipsoid separation metric."""
    dist_x = dfc.x_x - dfc.x_y
    dist_y = dfc.y_x - dfc.y_y
    dist_z = dfc.h_x - dfc.h_y

    ellipsoid = (dist_x / Rxy) ** 2 + (dist_y / Rxy) ** 2 + (dist_z / Rz) ** 6
    return ellipsoid


# %%

# Single-aircraft optimization
print("Running single-aircraft optimization...")
optimizer_single = top.Cruise(
    scenarios=scenarios, conflict=False, debug=True, max_nodes=30
)
df_s = optimizer_single.trajectory(objective="fuel")
# %%
# Multi-aircraft optimization
print("Running multi-aircraft optimization...")
optimizer_multi = top.Cruise(
    scenarios=scenarios, conflict=True, debug=True, max_nodes=30, max_iterations=1000
)
df_m = optimizer_multi.trajectory(objective="fuel", initial_guess=df_s)

# %%


print("\n" + "=" * 60)
print("FUEL BURN COMPARISON")
print("=" * 60)

pr = [
    {
        "multi ac1": np.round(df_m[0].mass.iloc[0] - df_m[0].mass.iloc[-1], 2),
        "multi ac2": np.round(df_m[1].mass.iloc[0] - df_m[1].mass.iloc[-1], 2),
        "single ac1": np.round(df_s[0].mass.iloc[0] - df_s[0].mass.iloc[-1], 2),
        "single ac2": np.round(df_s[1].mass.iloc[0] - df_s[1].mass.iloc[-1], 2),
        "multi sum": np.round(
            df_m[0].mass.iloc[0]
            - df_m[0].mass.iloc[-1]
            + df_m[1].mass.iloc[0]
            - df_m[1].mass.iloc[-1],
            6,
        ),
        "single sum": np.round(
            df_s[0].mass.iloc[0]
            - df_s[0].mass.iloc[-1]
            + df_s[1].mass.iloc[0]
            - df_s[1].mass.iloc[-1],
            6,
        ),
    }
]

results_df = pd.DataFrame(pr)
print(results_df)

# %%

print("\nPlotting multi-aircraft trajectories...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
for i in range(len(df_m)):
    ax1.plot(df_m[i].ts, df_m[i].longitude)
    ax2.plot(df_m[i].ts, df_m[i].latitude)
    ax3.plot(df_m[i].ts, df_m[i].altitude)

    ax1.scatter(df_m[i].ts, df_m[i].longitude, s=3)
    ax2.scatter(df_m[i].ts, df_m[i].latitude, s=3)
    ax3.scatter(df_m[i].ts, df_m[i].altitude, s=3)

ax1.set_ylabel("lon")
ax2.set_ylabel("lat")
ax3.set_ylabel("altitude")
ax3.set_xlabel("t")
plt.suptitle("Multi-aircraft optimization")
plt.show()

# %%

print("Plotting single-aircraft trajectories...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(df_s[0].ts, df_s[0].longitude)
ax1.plot(df_s[1].ts, df_s[1].longitude)
ax2.plot(df_s[0].ts, df_s[0].latitude)
ax2.plot(df_s[1].ts, df_s[1].latitude)
ax3.plot(df_s[0].ts, df_s[0].altitude)
ax3.plot(df_s[1].ts, df_s[1].altitude)

ax1.scatter(df_s[0].ts, df_s[0].longitude, s=3)
ax1.scatter(df_s[1].ts, df_s[1].longitude, s=3)
ax2.scatter(df_s[0].ts, df_s[0].latitude, s=3)
ax2.scatter(df_s[1].ts, df_s[1].latitude, s=3)
ax3.scatter(df_s[0].ts, df_s[0].altitude, s=3)
ax3.scatter(df_s[1].ts, df_s[1].altitude, s=3)

ax1.set_ylabel("lon")
ax2.set_ylabel("lat")
ax3.set_ylabel("altitude")
ax3.set_xlabel("t")
plt.suptitle("Single-aircraft optimization")
plt.show()

# %%

print("\n" + "=" * 60)
print("CONFLICT ANALYSIS - MULTI-AIRCRAFT")
print("=" * 60)
dfc_multi, mask_multi = analyze_conflict(
    df_m[0].copy(), df_m[1].copy(), Rxy, Rz, title="Conflict detection (multi-aircraft)"
)

ellipsoid_multi = compute_ellipsoid(dfc_multi, Rxy, Rz)
print(f"Min ellipsoid value: {ellipsoid_multi.min():.4f}")
print(f"Conflicts detected: {len(mask_multi)}")
# %%
print("\n" + "=" * 60)
print("CONFLICT ANALYSIS - SINGLE-AIRCRAFT")
print("=" * 60)
dfc_single, mask_single = analyze_conflict(
    df_s[0].copy(),
    df_s[1].copy(),
    Rxy,
    Rz,
    title="Conflict detection (single-aircraft)",
)
ellipsoid_single = compute_ellipsoid(dfc_single, Rxy, Rz)
print(f"Min ellipsoid value: {ellipsoid_single.min():.4f}")
print(f"Conflicts detected: {len(mask_single)}")

# %%

print("\n" + "=" * 60)
print("ANALYSIS WITH CUSTOM SEPARATION STANDARDS")
print("=" * 60)

Rxy_custom = 5 * openap.aero.nm
Rz_custom = 1000 * openap.aero.ft

ellipsoid_custom = compute_ellipsoid(dfc_single, Rxy_custom, Rz_custom)
print(f"Custom Rxy: {Rxy_custom / openap.aero.nm:.2f} nm")
print(f"Custom Rz: {Rz_custom / openap.aero.ft:.0f} ft")
print(f"Min ellipsoid value: {ellipsoid_custom.min():.4f}")
# %%plot on map
plot_trajectories_map(
    df_m, scenarios, title="Multiple-aircraft optimization (map)", color_by=None
)

# %%
plot_trajectories_map(df_s, scenarios, title="Single-aircraft optimization (map)")
# %%
