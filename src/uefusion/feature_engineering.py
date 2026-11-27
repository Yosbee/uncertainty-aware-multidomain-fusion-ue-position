"""Historical row/vehicle eligibility and non-coordinate feature derivations.

Retain published vehicle roles from configs/vehicle_roles.json. This module does
not redraw the split or the ten preselected demonstration vehicles.
"""
import numpy as np
import pandas as pd
from .observations import BS as BS_NAMES
from .cell_state import BSXY
BS_POSITIONS = dict(zip(BS_NAMES, map(tuple, BSXY)))
FILTER_CONFIG = {'min_best_bs_idx':0, 'min_speed_threshold':.1, 'train_min_dwell_time':60, 'train_min_avg_speed':1., 'train_min_mobility_ratio':.4}

def filter_step_0a(df: pd.DataFrame) -> pd.DataFrame:
    """Step 0-A: Remove bad data (no coverage + complete stops)."""
    print("\n" + "=" * 70)
    print("STEP 0-A: Remove bad data")
    print("=" * 70)

    initial_size = len(df)

    # Remove no-coverage samples
    mask_coverage = df['best_bs_idx'] >= FILTER_CONFIG['min_best_bs_idx']
    removed_coverage = (~mask_coverage).sum()
    df = df[mask_coverage].copy()
    print(f"  Removed no-coverage (BS -1): {removed_coverage:,} ({removed_coverage/initial_size*100:.1f}%)")

    # Remove complete stops (speed < 0.1 m/s)
    # But keep sequences with brief stops (handled at sequence level later)
    mask_speed = df['speed'] >= FILTER_CONFIG['min_speed_threshold']
    removed_speed = (~mask_speed).sum()
    df = df[mask_speed].copy()
    print(f"  Removed complete stops (<0.1 m/s): {removed_speed:,} ({removed_speed/initial_size*100:.1f}%)")

    print(f"  Remaining: {len(df):,} samples ({len(df)/initial_size*100:.1f}%)")
    print(f"  Remaining vehicles: {df['vehicle_id'].nunique():,}")

    return df

def compute_vehicle_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-vehicle statistics for filtering decisions."""
    print("\nComputing per-vehicle statistics...")

    # Sort for consistent processing
    df = df.sort_values(['vehicle_id', 'timestamp'])

    # Compute handovers (changes in best_bs_idx)
    df['bs_changed'] = df.groupby('vehicle_id')['best_bs_idx'].diff().ne(0).astype(int)
    # First sample of each vehicle is not a handover
    df.loc[df.groupby('vehicle_id').head(1).index, 'bs_changed'] = 0

    # Per-vehicle aggregation
    vehicle_stats = df.groupby('vehicle_id').agg(
        dwell_time=('timestamp', lambda x: x.max() - x.min()),
        n_samples=('timestamp', 'count'),
        avg_speed=('speed', 'mean'),
        max_speed=('speed', 'max'),
        mobility_ratio=('speed', lambda x: (x > 0.5).mean()),
        n_handovers=('bs_changed', 'sum'),
        n_unique_bs=('best_bs_idx', 'nunique'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
    ).reset_index()

    print(f"  Computed stats for {len(vehicle_stats):,} vehicles")

    return vehicle_stats

def filter_step_0b(df: pd.DataFrame, vehicle_stats: pd.DataFrame) -> tuple:
    """Step 0-B: Select training vehicles."""
    print("\n" + "=" * 70)
    print("STEP 0-B: Training vehicle selection")
    print("=" * 70)

    config = FILTER_CONFIG

    mask = (
        (vehicle_stats['dwell_time'] >= config['train_min_dwell_time']) &
        (vehicle_stats['avg_speed'] > config['train_min_avg_speed']) &
        (vehicle_stats['mobility_ratio'] > config['train_min_mobility_ratio'])
    )

    selected_vehicles = vehicle_stats[mask]['vehicle_id'].values
    rejected_vehicles = vehicle_stats[~mask]['vehicle_id'].values

    print(f"  Criteria: dwell >= {config['train_min_dwell_time']}s, "
          f"avg_speed > {config['train_min_avg_speed']} m/s, "
          f"mobility > {config['train_min_mobility_ratio']*100:.0f}%")
    print(f"  Selected: {len(selected_vehicles):,} vehicles "
          f"({len(selected_vehicles)/len(vehicle_stats)*100:.1f}%)")
    print(f"  Rejected: {len(rejected_vehicles):,} vehicles")

    # Filter data
    df_filtered = df[df['vehicle_id'].isin(selected_vehicles)].copy()
    print(f"  Filtered samples: {len(df_filtered):,}")

    # Stats on selected vehicles
    selected_stats = vehicle_stats[mask]
    print(f"\n  Selected vehicle stats:")
    print(f"    Avg dwell time: {selected_stats['dwell_time'].mean():.0f}s")
    print(f"    Avg speed: {selected_stats['avg_speed'].mean():.1f} m/s")
    print(f"    Avg handovers: {selected_stats['n_handovers'].mean():.1f}")

    return df_filtered, selected_vehicles

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features needed by all three models."""
    print("\nComputing derived features...")
    df = df.sort_values(['vehicle_id', 'timestamp'])

    # --- Serving cell one-hot ---
    for i, bs_name in enumerate(BS_NAMES):
        df[f'serving_cell_{i}'] = (df['best_bs_idx'] == i).astype(np.float32)

    # --- Cell centroid (serving BS position) ---
    df['cell_centroid_x'] = 0.0
    df['cell_centroid_y'] = 0.0
    for i, bs_name in enumerate(BS_NAMES):
        mask = df['best_bs_idx'] == i
        df.loc[mask, 'cell_centroid_x'] = BS_POSITIONS[bs_name][0]
        df.loc[mask, 'cell_centroid_y'] = BS_POSITIONS[bs_name][1]

    # --- Handover detection ---
    df['bs_changed'] = df.groupby('vehicle_id')['best_bs_idx'].diff().ne(0).astype(int)
    df.loc[df.groupby('vehicle_id').head(1).index, 'bs_changed'] = 0
    df['handover_flag'] = df['bs_changed']

    # --- Time since handover ---
    df['time_since_handover'] = 0.0
    for vid, vdf in df.groupby('vehicle_id'):
        ho_mask = vdf['handover_flag'] == 1
        ho_times = vdf.loc[ho_mask, 'timestamp'].values
        if len(ho_times) > 0:
            for idx, row in vdf.iterrows():
                past_ho = ho_times[ho_times <= row['timestamp']]
                if len(past_ho) > 0:
                    df.loc[idx, 'time_since_handover'] = row['timestamp'] - past_ho[-1]
    # Cap at 300s to avoid extreme values
    df['time_since_handover'] = df['time_since_handover'].clip(0, 300)

    # --- Handover direction vector ---
    df['handover_dir_x'] = 0.0
    df['handover_dir_y'] = 0.0
    # prev BS centroid
    df['prev_bs_idx'] = df.groupby('vehicle_id')['best_bs_idx'].shift(1)
    for i, bs_name in enumerate(BS_NAMES):
        mask = (df['handover_flag'] == 1) & (df['prev_bs_idx'] == i)
        if mask.any():
            # direction = current BS centroid - previous BS centroid
            df.loc[mask, 'handover_dir_x'] = df.loc[mask, 'cell_centroid_x'] - BS_POSITIONS[bs_name][0]
            df.loc[mask, 'handover_dir_y'] = df.loc[mask, 'cell_centroid_y'] - BS_POSITIONS[bs_name][1]
    # Forward-fill handover direction within each vehicle
    df['handover_dir_x'] = df.groupby('vehicle_id')['handover_dir_x'].transform(
        lambda x: x.replace(0, np.nan).ffill().fillna(0)
    )
    df['handover_dir_y'] = df.groupby('vehicle_id')['handover_dir_y'].transform(
        lambda x: x.replace(0, np.nan).ffill().fillna(0)
    )

    # --- Cell dwell time (continuous time in current cell) ---
    df['cell_dwell_time'] = 0.0
    for vid, vdf in df.groupby('vehicle_id'):
        dwell = 0.0
        prev_ts = None
        prev_bs = None
        for idx, row in vdf.iterrows():
            if prev_bs is not None and row['best_bs_idx'] == prev_bs and prev_ts is not None:
                dt = row['timestamp'] - prev_ts
                dwell += dt
            else:
                dwell = 0.0
            df.loc[idx, 'cell_dwell_time'] = dwell
            prev_ts = row['timestamp']
            prev_bs = row['best_bs_idx']
    df['cell_dwell_time'] = df['cell_dwell_time'].clip(0, 600)

    # --- Delta speed ---
    df['delta_speed'] = df.groupby('vehicle_id')['speed'].diff().fillna(0).clip(-10, 10)

    # --- Radio derived features ---
    # RSS from all 6 BS
    rss_cols = [f'rss_bs_{bs}' for bs in BS_NAMES]
    sinr_cols = [f'sinr_bs_{bs}' for bs in BS_NAMES]

    # Radio changes for all BS
    for col in rss_cols + sinr_cols:
        if col in df.columns:
            df[f'd_{col}'] = df.groupby('vehicle_id')[col].diff().fillna(0).clip(-30, 30)

    # RSS range (max - min across all BS)
    if all(col in df.columns for col in rss_cols):
        rss_values = df[rss_cols].values
        df['rss_range'] = rss_values.max(axis=1) - rss_values.min(axis=1)

        # RSS margin (best - 2nd best)
        sorted_rss = np.sort(rss_values, axis=1)[:, ::-1]
        df['rss_margin'] = sorted_rss[:, 0] - sorted_rss[:, 1]

    if all(col in df.columns for col in sinr_cols):
        sinr_values = df[sinr_cols].values
        df['sinr_range'] = sinr_values.max(axis=1) - sinr_values.min(axis=1)

    # Clean up temp columns
    df.drop(columns=['bs_changed', 'prev_bs_idx'], errors='ignore', inplace=True)

    print(f"  Added derived features. Total columns: {len(df.columns)}")

    return df
