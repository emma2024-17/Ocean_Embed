import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st
from urllib.error import URLError
from urllib.request import Request, urlopen
from train_model import OceanEmbedUNet, load_and_preprocess_nc

# ==========================================
# GOOGLE DRIVE DATASET SETUP (additive only)
# ==========================================
# This block downloads the .nc dataset from Google Drive the first time the
# app runs, and does nothing on later runs since the file already exists.
# It does not touch or override any of the logic below.
GDRIVE_FILE_ID = "1stCmdQEnUZnmyyzlLss8O1JtK42Zknpu"
NC_DATA_PATH = "./data/glorys_subset.nc"


@st.cache_data(show_spinner="Downloading ocean dataset from Google Drive...")
def ensure_dataset_downloaded():
    """Download the GLORYS subset from Google Drive if it isn't already present
    locally at the path the rest of the app expects (./data/glorys_subset.nc)."""
    os.makedirs(os.path.dirname(NC_DATA_PATH), exist_ok=True)
    if not os.path.exists(NC_DATA_PATH):
        import gdown  # imported here so the app still runs if gdown isn't installed
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            NC_DATA_PATH,
            quiet=False,
        )
    return NC_DATA_PATH


ensure_dataset_downloaded()
# ==========================================
# YOUR EXISTING CODE STARTS BELOW, UNCHANGED
# ==========================================

st.set_page_config(layout="wide", page_title="OceanEmbed PoC Dashboard")

# Presentation-only styling: the model, controls, and inference flow remain unchanged.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --abyss:#041b2d; --foam:#dff8fa; --sand:#f5d79a; }
    .stApp { background:radial-gradient(circle at 84% 4%,rgba(22,181,197,.20),transparent 28rem),radial-gradient(circle at 16% 88%,rgba(8,126,139,.19),transparent 30rem),linear-gradient(145deg,var(--abyss),#062c45 46%,#064b63); color:var(--foam); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:rgba(4,27,45,.72); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#05283f,#06334c); border-right:1px solid rgba(149,232,238,.16); }
    [data-testid="stSidebar"] * { color:var(--foam); }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#fff !important; letter-spacing:-.025em; }
    h1 { text-shadow:0 0 28px rgba(59,211,225,.30); }
    [data-testid="stCaptionContainer"] { color:#a9dfe5 !important; }
    [data-testid="stMetric"] { background:linear-gradient(135deg,rgba(11,81,105,.86),rgba(5,51,76,.86)); border:1px solid rgba(140,231,237,.22); border-radius:14px; padding:.9rem 1rem; box-shadow:0 12px 26px rgba(0,17,32,.18); }
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"] { color:#ecfeff !important; }
    [data-testid="stMetricDelta"] svg { fill:var(--sand); }
    .stButton > button { width:100%; color:#032439 !important; background:linear-gradient(135deg,#64dce5,#26a8bc); border:none; border-radius:10px; font-weight:700; box-shadow:0 7px 18px rgba(0,0,0,.24); transition:transform 150ms ease,box-shadow 150ms ease; }
    .stButton > button:hover { color:#021925 !important; transform:translateY(-1px); box-shadow:0 10px 23px rgba(0,0,0,.30); }
    [data-baseweb="select"] > div,[data-baseweb="slider"] div[role="slider"] { background-color:#0a4e67 !important; border-color:rgba(161,235,239,.34) !important; }
    [data-testid="stDivider"] { border-color:rgba(160,232,236,.18); }
    /* The built-in spinner is too faint on this dark background. The custom
       processing banner below is the single, accessible run indicator. */
    [data-testid="stSpinner"] { display:none !important; }
    .processing-banner { display:flex; align-items:center; gap:14px; margin:.25rem 0 1.2rem; padding:1rem 1.2rem; border:1px solid rgba(128,233,239,.44); border-radius:14px; background:linear-gradient(100deg,rgba(12,99,125,.92),rgba(8,54,82,.92)); box-shadow:0 12px 28px rgba(0,13,29,.24); color:#edfeff; }
    .processing-orbit { width:26px; height:26px; box-sizing:border-box; border:3px solid rgba(191,249,250,.25); border-top-color:#9af6f3; border-right-color:#f5d79a; border-radius:50%; animation:ocean-spin .85s linear infinite; flex:0 0 auto; }
    .processing-title { font-family:'Space Grotesk',sans-serif; font-size:1.02rem; font-weight:700; }
    .processing-detail { color:#b8e8ec; font-size:.88rem; margin-top:2px; }
    @keyframes ocean-spin { to { transform:rotate(360deg); } }
    </style>
    """, unsafe_allow_html=True,
)

plt.rcParams.update({"figure.facecolor":"#073b57", "axes.facecolor":"#083f59", "axes.edgecolor":"#91dce2", "axes.labelcolor":"#e5fbfc", "xtick.color":"#c3edf0", "ytick.color":"#c3edf0", "text.color":"#ecfeff", "grid.color":"#4a8d9c"})

st.title("🌊 OceanEmbed: Subsurface Ocean Temperature Reconstruction")
st.caption("Proof-of-Concept Dashboard | Ministry of Earth Sciences (MoES) - INCOIS | Problem Statement ID-26066")

DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def fallback_ocean_region(latitude, longitude):
    """Return an honest local label when the Marine Regions service is offline.

    The coordinate controls are restricted to the Indian Ocean data subset, so
    this is a useful description without claiming an unverified IHO boundary.
    """
    return f"Indian Ocean ({latitude:.2f}\N{DEGREE SIGN}N, {longitude:.2f}\N{DEGREE SIGN}E)", None


@st.cache_data(ttl=86_400, show_spinner=False)
def get_ocean_region(latitude, longitude):
    """Look up the authoritative International Hydrographic Organization (IHO)
    sea area containing a coordinate.

    Marine Regions performs the underlying point-in-polygon operation against
    the IHO *Limits of Oceans and Seas* dataset. If that online service is not
    reachable, use a clearly labelled local description for this app's Indian
    Ocean-only coordinate range.
    """
    url = (
        "https://www.marineregions.org/rest/"
        f"getGazetteerRecordsByLatLong.json/{latitude:.6f}/{longitude:.6f}/"
    )
    try:
        request = Request(url, headers={"User-Agent": "OceanEmbed/1.0"})
        with urlopen(request, timeout=8) as response:
            records = json.load(response)
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        return fallback_ocean_region(latitude, longitude)

    iho_areas = [
        record for record in records
        if record.get("placeType") == "IHO Sea Area"
        and record.get("preferredGazetteerNameLang") == "English"
    ]
    if not iho_areas:
        return fallback_ocean_region(latitude, longitude)

    # A coordinate can belong to a broad ocean and a nested sea. Select the
    # smallest matching IHO polygon to report the most specific valid name.
    def area_size(record):
        values = ("minLatitude", "maxLatitude", "minLongitude", "maxLongitude")
        if any(record.get(value) is None for value in values):
            return float("inf")
        return (record["maxLatitude"] - record["minLatitude"]) * (
            record["maxLongitude"] - record["minLongitude"]
        )

    region = min(iho_areas, key=area_size)
    return region["preferredGazetteerName"], region.get("MRGID")

@st.cache_resource
def load_trained_model():
    model = OceanEmbedUNet()
    t_min, t_max = 3.5, 32.0
    if os.path.exists("oceanembed_model.pth"):
        checkpoint = torch.load("oceanembed_model.pth", map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            t_min = checkpoint.get('t_min', 3.5)
            t_max = checkpoint.get('t_max', 32.0)
    model.eval()
    return model, t_min, t_max

model, t_min, t_max = load_trained_model()

st.sidebar.header("🕹️ Controls & Coordinates")

selected_depth = st.sidebar.select_slider(
    "Select Output Depth Layer",
    options=DEPTHS,
    value=1000,
    format_func=lambda x: f"{x} meters"
)
selected_depth_idx = DEPTHS.index(selected_depth)

# Geographic coordinate selectors
selected_lat = st.sidebar.slider("Latitude (°N)", 5.0, 30.0, 12.0, step=0.25)
selected_lon = st.sidebar.slider("Longitude (°E)", 45.0, 105.0, 65.0, step=0.25)

sidebar_region, _ = get_ocean_region(selected_lat, selected_lon)
st.sidebar.caption(f"Region: **{sidebar_region}**")
run_button = st.sidebar.button("Run 3D Subsurface Reconstruction Pipeline")

if run_button:
    data_path = "./data/glorys_subset.nc"
    
    if not os.path.exists(data_path):
        st.error("Dataset not found! Please ensure glorys_subset.nc exists in ./data/")
    else:
        processing_notice = st.empty()
        processing_notice.markdown(
            """
            <div class="processing-banner">
                <div class="processing-orbit"></div>
                <div><div class="processing-title">Executing 3D Neural Inference…</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.spinner("Executing 3D Neural Inference..."):
            inputs, targets, ocean_mask, targets_raw, lats, lons = load_and_preprocess_nc(data_path)
            with torch.no_grad():
                preds_norm, latent_embed = model(inputs)
                
            preds_denorm = preds_norm.numpy() * (t_max - t_min) + t_min
            processing_notice.empty()
            
            # Map selected geographic lat/lon to closest array index
            lat_idx = int(np.abs(lats - selected_lat).argmin())
            lon_idx = int(np.abs(lons - selected_lon).argmin())
            selected_region, _ = get_ocean_region(lats[lat_idx], lons[lon_idx])

            mask_np = ocean_mask.numpy()
            pred_map = np.where(mask_np, preds_denorm[0, selected_depth_idx], np.nan)
            gt_map = np.where(mask_np, targets_raw[0, selected_depth_idx], np.nan)
            
            pred_valid = pred_map[mask_np]
            gt_valid = gt_map[mask_np]
            
            valid_pixels = ~np.isnan(pred_valid) & ~np.isnan(gt_valid)
            
            rmse = float(np.sqrt(np.mean((pred_valid[valid_pixels] - gt_valid[valid_pixels]) ** 2)))
            bias = float(np.mean(pred_valid[valid_pixels] - gt_valid[valid_pixels]))
            corr = float(np.corrcoef(pred_valid[valid_pixels], gt_valid[valid_pixels])[0, 1]) if np.sum(valid_pixels) > 0 else 0.0
            
            # Extract specific predicted and actual values at selected star coordinates
            if mask_np[lat_idx, lon_idx]:
                pt_pred_temp = float(pred_map[lat_idx, lon_idx])
                pt_gt_temp = float(gt_map[lat_idx, lon_idx])
                pred_str = f"{pt_pred_temp:.2f} °C"
                gt_str = f"{pt_gt_temp:.2f} °C"
                # Use the values visible in the cards, so 8.38 vs 8.40
                # consistently displays a -0.02 °C difference.
                displayed_pred = float(f"{pt_pred_temp:.2f}")
                displayed_gt = float(f"{pt_gt_temp:.2f}")
                diff_val = displayed_pred - displayed_gt
                diff_str = f"{diff_val:+.2f} °C"
            else:
                pred_str = "Land"
                gt_str = "Land"
                diff_str = "N/A"

            # Top metric row displaying predicted and target temperatures at selected point
            col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
            col_m1.metric("Selected Depth", f"{selected_depth} m")
            col_m2.metric("AI Pred Temp (★)", pred_str, delta=diff_str, delta_color="inverse")
            col_m3.metric("Actual Temp (★)", gt_str)
            col_m4.metric("RMSE Error", f"{rmse:.3f} °C")
            col_m5.metric("Bias", f"{bias:.3f} °C")
            col_m6.metric("Pearson Corr", f"{corr:.3f}")
            st.caption(
                f"Selected region: **{selected_region}** · "
                f"{lats[lat_idx]:.2f}°N, {lons[lon_idx]:.2f}°E"
            )
            
            st.divider()
            
            vmin_val = float(np.nanmin(gt_map))
            vmax_val = float(np.nanmax(gt_map))
            extent = [lons.min(), lons.max(), lats.min(), lats.max()]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"AI Reconstructed Map ({selected_depth}m)")
                fig1, ax1 = plt.subplots(figsize=(6, 3.5))
                c1 = ax1.imshow(pred_map, cmap='coolwarm', origin='lower', extent=extent, vmin=vmin_val, vmax=vmax_val)
                ax1.plot(lons[lon_idx], lats[lat_idx], marker='*', markersize=15, color='red', label=selected_region)
                ax1.annotate(selected_region, (lons[lon_idx], lats[lat_idx]), xytext=(8, 8), textcoords="offset points", color="white", weight="bold")
                ax1.legend(loc="best")
                ax1.set_xlabel("Longitude (°E)")
                ax1.set_ylabel("Latitude (°N)")
                fig1.colorbar(c1, ax=ax1, label="Temp (°C)")
                st.pyplot(fig1)
                
            with col2:
                st.subheader(f"Ground Truth GLORYS Map ({selected_depth}m)")
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                c2 = ax2.imshow(gt_map, cmap='coolwarm', origin='lower', extent=extent, vmin=vmin_val, vmax=vmax_val)
                ax2.plot(lons[lon_idx], lats[lat_idx], marker='*', markersize=15, color='red', label=selected_region)
                ax2.annotate(selected_region, (lons[lon_idx], lats[lat_idx]), xytext=(8, 8), textcoords="offset points", color="white", weight="bold")
                ax2.legend(loc="best")
                ax2.set_xlabel("Longitude (°E)")
                ax2.set_ylabel("Latitude (°N)")
                fig2.colorbar(c2, ax=ax2, label="Temp (°C)")
                st.pyplot(fig2)

            st.divider()

            st.subheader("3D AI Ocean Temperature Volume")
            st.caption("Each point is a reconstructed water-column sample; colour represents temperature.")

            # Thin the latitude/longitude grid so the 3D plot remains responsive
            # while retaining every reconstructed depth layer.
            sample_step = max(1, int(np.ceil(max(mask_np.shape) / 28)))
            sample_lats = lats[::sample_step]
            sample_lons = lons[::sample_step]
            sample_mask = mask_np[::sample_step, ::sample_step]
            volume_temp = preds_denorm[0, :, ::sample_step, ::sample_step]
            depth_grid, lat_grid, lon_grid = np.meshgrid(
                np.asarray(DEPTHS), sample_lats, sample_lons, indexing="ij"
            )
            volume_valid = np.broadcast_to(sample_mask, volume_temp.shape)

            fig_volume = plt.figure(figsize=(11, 5.5))
            ax_volume = fig_volume.add_subplot(111, projection="3d")
            volume_colours = ax_volume.scatter(
                lon_grid[volume_valid], lat_grid[volume_valid], -depth_grid[volume_valid],
                c=volume_temp[volume_valid], cmap="coolwarm", s=10, alpha=0.72,
                vmin=t_min, vmax=t_max, linewidths=0,
            )
            if mask_np[lat_idx, lon_idx]:
                ax_volume.plot(
                    np.full(len(DEPTHS), lons[lon_idx]),
                    np.full(len(DEPTHS), lats[lat_idx]), -np.asarray(DEPTHS),
                    color="#fff3a3", linewidth=2, label=f"{selected_region} column",
                )
                ax_volume.scatter(
                    [lons[lon_idx]], [lats[lat_idx]], [-selected_depth],
                    color="#ff4f70", marker="*", s=140, depthshade=False,
                )
                ax_volume.legend(loc="upper left", fontsize=8)
            ax_volume.set_xlabel("Longitude (°E)", labelpad=8)
            ax_volume.set_ylabel("Latitude (°N)", labelpad=8)
            ax_volume.set_zlabel("Depth (m)", labelpad=8)
            ax_volume.view_init(elev=24, azim=-128)
            fig_volume.colorbar(volume_colours, ax=ax_volume, pad=0.08, shrink=0.65, label="Temperature (°C)")
            fig_volume.tight_layout()
            st.pyplot(fig_volume)

            st.divider()

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("🧠 Satellite Latent Embedding Space")
                embed_map = np.where(mask_np[::4, ::4], latent_embed[0, 0].numpy(), np.nan)
                fig3, ax3 = plt.subplots(figsize=(6, 3.5))
                c3 = ax3.imshow(embed_map, cmap='viridis', origin='lower', extent=extent)
                ax3.set_xlabel("Longitude (°E)")
                ax3.set_ylabel("Latitude (°N)")
                fig3.colorbar(c3, ax=ax3, label="Activation")
                st.pyplot(fig3)
                
            with col4:
                st.caption(f"Region: {selected_region}")
                st.subheader(f"📊 Profile at ({lats[lat_idx]:.2f}°N, {lons[lon_idx]:.2f}°E)")
                
                if not mask_np[lat_idx, lon_idx]:
                    st.warning("⚠️ Selected location is on LAND mass. Adjust Latitude / Longitude sliders to an ocean coordinate.")
                else:
                    pred_profile = preds_denorm[0, :, lat_idx, lon_idx]
                    gt_profile = targets_raw[0, :, lat_idx, lon_idx]
                    
                    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
                    ax4.plot(pred_profile, DEPTHS, 'o-', label='AI Predicted')
                    ax4.plot(gt_profile, DEPTHS, 's--', label='GLORYS Target')
                    ax4.invert_yaxis()
                    ax4.set_xlabel("Temperature (°C)")
                    ax4.set_ylabel("Depth (m)")
                    ax4.legend(loc='best')
                    st.pyplot(fig4)
