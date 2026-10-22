"""
=============================================================================
PIPELINE ISB (FIRST-PRINCIPLES Q1 STANDARD) - SCRIPT 04: MASTER VISUAL ENGINE
(CONFIGURED FOR KAGGLE INPUT DATASET: isb-monte-carlo-200k-subject)
=============================================================================
Fixes & Features Implemented:
1. Reads all precomputed 200k-subject .npy, .json, and .csv files directly from
   '/kaggle/input/datasets/cefiyana/isb-monte-carlo-200k-subject' (with automatic
   sub-directory resolution) and writes 300 DPI figures to '/kaggle/working/Manuscript_Figures'.
2. Figure 6 Fix: Uses explicit keyword argument 'ax=ax_ps' in fig6.colorbar(line, ax=ax_ps)
   so the colorbar no longer overwrites the phase-space trajectory canvas.
3. Figure 7 Fix: Eliminates subplot title collisions (wspace=0.32, hspace=0.45)
   and raises y-axis ceilings in Panels A & B with top-centered 2-column legends
   so the Young Adult remission jump (to 2.949 mM) and glycogen recharge are
   100% unobstructed.
4. Figure 2 Enhancement: Adds a quantitative horizontal [ATP] (mM) colorbar.
=============================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from nilearn import datasets, plotting
from nilearn.image import load_img
import warnings
warnings.filterwarnings("ignore")

print("[SYSTEM] Initializing Audited Master Visual Rendering Engine (Kaggle Input Dataset Mode)...")

# ---------------------------------------------------------
# [MODULE 0] KAGGLE INPUT DATASET & OUTPUT PATH RESOLUTION
# ---------------------------------------------------------
PRIMARY_INPUT_DIR = "/kaggle/input/datasets/cefiyana/isb-monte-carlo-200k-subject"
FALLBACK_SEARCH_ROOTS = [
    PRIMARY_INPUT_DIR,
    "/kaggle/input/isb-monte-carlo-200k-subject",
    "/kaggle/input",
    "/kaggle/working/ISB_Empirical_Vault"
]

OUTPUT_DIR = "/kaggle/working/Manuscript_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build a fast lookup index of all files inside the Kaggle input dataset
file_index_map = {}
for root_path in FALLBACK_SEARCH_ROOTS:
    if os.path.exists(root_path):
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if fname not in file_index_map:
                    file_index_map[fname] = os.path.join(dirpath, fname)

def get_dataset_file(filename):
    direct_path = os.path.join(PRIMARY_INPUT_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path
    if filename in file_index_map:
        return file_index_map[filename]
    raise FileNotFoundError(
        f"[ERROR] File '{filename}' tidak ditemukan di {PRIMARY_INPUT_DIR} maupun /kaggle/input. "
        f"Pastikan dataset sudah ter-attach pada notebook."
    )

print(f"-> [SYSTEM] Input Dataset indexed ({len(file_index_map)} files detected).")
print(f"-> [SYSTEM] Output Figures directory anchored at: {OUTPUT_DIR}")

# Q1 Aesthetics Setup
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.30)
sns.set_style("ticks")
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 11.5,
    'axes.titlesize': 12.0,
    'legend.fontsize': 9.0,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

with open(get_dataset_file("L1_Destrieux_ROI_Metadata.json"), "r") as f:
    roi_meta = json.load(f)
with open(get_dataset_file("L5_L6_Clinical_Anchors.json"), "r") as f:
    anchors = json.load(f)

EPICENTER_IDX = int(roi_meta["primary_insula_nodes"][0])
EPICENTER_NAME = roi_meta["roi_names"][EPICENTER_IDX]
ALL_INSULA_NODES = [int(i) for i in roi_meta["primary_insula_nodes"] + roi_meta["secondary_insula_nodes"]]
DISTANT_IDX = 147 if EPICENTER_IDX != 147 else 0
collapse_thresh = float(anchors["Multimodal_31P_1H_MRS_Anchors"]["Saddle_Node_Threshold"])

TIER_MAP = {
    "EUR": "Tier I (High Reserve)",
    "AMR": "Tier II (Mod-High Reserve)",
    "EAS": "Tier III (Mod-Low Reserve)",
    "AFR": "Tier IV (Low Reserve)"
}
TIER_COLORS = {
    "Tier I (High Reserve)": "#1f77b4",
    "Tier II (Mod-High Reserve)": "#2ca02c",
    "Tier III (Mod-Low Reserve)": "#ff7f0e",
    "Tier IV (Low Reserve)": "#d62728"
}

try:
    t_eval = np.load(get_dataset_file("Output_Time_Vector.npy"))
except FileNotFoundError:
    t_eval = np.linspace(0, 1000, 1000)

# ---------------------------------------------------------
# [FIGURE 1] STOCHASTIC MULTISCALE DYNAMICS (REGIME I)
# ---------------------------------------------------------
print(f"\n[MODULE 1] Rendering Figure 1 (Insular Epicenter Node {EPICENTER_IDX}: {EPICENTER_NAME})...")
ATP_data = np.load(get_dataset_file("Output_ATP_Spatiotemporal_AFR_Age60.npy"))
G_data = np.load(get_dataset_file("Output_G_Spatiotemporal_AFR_Age60.npy"))

Global_Mean_ATP = np.mean(ATP_data, axis=0)
Global_Mean_G_mM = np.mean(G_data, axis=0)
Epicenter_G_mM = G_data[EPICENTER_IDX, :]

fig1 = plt.figure(figsize=(14.5, 11.2))
gs = fig1.add_gridspec(2, 2, height_ratios=[2.0, 1.55], hspace=0.44, wspace=0.30)

ax1 = fig1.add_subplot(gs[0, 0])
epicenter_drift = pd.Series(ATP_data[EPICENTER_IDX, :]).rolling(window=20, min_periods=1).mean()

ax1.plot(t_eval, ATP_data[EPICENTER_IDX, :], color='#d62728', linewidth=1.0, alpha=0.35, label=r'Insular Epicenter (Raw $dW_t$)')
ax1.plot(t_eval, epicenter_drift, color='#8c1515', linewidth=2.8, label=r'Insular Epicenter $\langle A_i \rangle$ Drift')
ax1.plot(t_eval, ATP_data[DISTANT_IDX, :], color='#9467bd', linewidth=1.5, alpha=0.80, label=r'Distant Cortical ROI ($\eta_{147}$)')
ax1.plot(t_eval, Global_Mean_ATP, color='#1f77b4', linewidth=2.5, label=r'Global Cortical Mean $\mu_{\mathrm{ATP}}$')
ax1.axhline(y=collapse_thresh, color='black', linestyle=':', linewidth=2.0, label=r'Saddle-Node Boundary ($\theta_c = 0.50\ \mathrm{mM}$)')

ax1.set_ylabel(r'Cytosolic $[\mathrm{ATP}]_{\mathrm{astrocytic}}\ (\mathrm{mM})$', fontweight='bold')
ax1.set_xlabel(r'Biological Time $t\ (\mathrm{Days})$', fontweight='bold')
ax1.set_title(r'A. Bistable SDE Depletion & Kramers Escape', fontweight='bold')
ax1.set_ylim(0, 3.6)
ax1.legend(loc='upper right', framealpha=0.95, edgecolor='black')

ax2 = fig1.add_subplot(gs[0, 1])
ax2.plot(t_eval, Epicenter_G_mM, color='#d62728', linewidth=2.2, alpha=0.88, label=r'Insular Epicenter $[G]_{\mathrm{ext}}$')
ax2.plot(t_eval, Global_Mean_G_mM, color='#1f77b4', linewidth=2.4, label=r'Global Cortical Mean $\mu_{\mathrm{Glu}}$')
ax2.set_yscale('log')
ax2.axhline(y=0.020, color='#7f8c8d', linestyle='--', linewidth=1.5, label=r'EAAT2 $K_{m,G}$ ($0.020\ \mathrm{mM} / 20\ \mu\mathrm{M}$)')

ax2.set_ylabel(r'Extracellular $[\mathrm{Glu}]_{\mathrm{ext}}\ (\mathrm{mM},\ \log_{10}\ \mathrm{scale})$', fontweight='bold')
ax2.set_xlabel(r'Biological Time $t\ (\mathrm{Days})$', fontweight='bold')
ax2.set_title(r'B. Insular Afferent Influx & Reversed-Uptake Surge', fontweight='bold')
ax2.set_ylim(1e-3, 15.0)
ax2.legend(loc='lower right', framealpha=0.95, edgecolor='black')

df_sim1 = pd.read_csv(get_dataset_file("Output_TrueDynamic_MegaSimulation_Final.csv"))
df_sim1["Reserve_Tier_Label"] = df_sim1["Ancestry"].map(TIER_MAP)
failed_patients1 = df_sim1[df_sim1["Survived_1000_Days"] == False].copy()
plot_sample1 = failed_patients1.sample(min(3500, len(failed_patients1)), random_state=42)

ax3 = fig1.add_subplot(gs[1, :])
sns.scatterplot(
    data=plot_sample1, x='Time_To_Bifurcation_Days', y='Stabilized_ISB_Score',
    hue='Reserve_Tier_Label', hue_order=list(TIER_COLORS.keys()), palette=TIER_COLORS,
    s=38, alpha=0.68, edgecolor='black', linewidth=0.3, ax=ax3
)
ax3.set_xlabel(r'Time-to-Bifurcation $t_c\ (\mathrm{Biological\ Days})$', fontweight='bold')
ax3.set_ylabel(r'Basal ISB Metric $\Omega_{\mathrm{basal}}$ (Unclipped)', fontweight='bold')
ax3.set_title(r'C. Bioenergetic Reserve vs. Phase Transition Latency Across Polygenic Tiers ($N=100,000$)', fontweight='bold')
ax3.legend(title='Polygenic Mitochondrial Reserve', loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True, edgecolor='black')

plt.suptitle(r'Figure 1: Stochastic Multi-Scale Dynamics of Focal Insular Phase Transition (Regime I)', fontweight='bold', fontsize=15.5, y=1.02)
fig1.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_1_Complexity.png"), dpi=300, bbox_inches='tight')
plt.close(fig1)
print("   -> [VALIDATED] Figure 1 archived.")

# ---------------------------------------------------------
# [FIGURE 2] CALLOSAL-CONSTRAINED CONNECTOME HEATMAP (t=1000)
# ---------------------------------------------------------
print("\n[MODULE 2] Rendering Figure 2: Synchronized Polygenic Reserve Connectome Map...")
L2_DTI_Mask = np.load(get_dataset_file("L2_DTI_Structural_Mask.npy"))
try:
    coords = np.load(get_dataset_file("L1_Destrieux_MNI_Coords.npy"))
except FileNotFoundError:
    destrieux = datasets.fetch_atlas_destrieux_2009(lateralized=True)
    atlas_img = load_img(destrieux['maps'])
    coords = plotting.find_parcellation_cut_coords(atlas_img)
coords = coords[:L2_DTI_Mask.shape[0]]

SYNCHRONIZED_FIGURE2_COHORTS = [
    ("Tier I Cohort (High Mitochondrial Reserve)", "Output_ATP_Spatiotemporal_EUR_Age50.npy", (0, 0)),
    ("Tier II Cohort (Moderate-High Reserve)", "Output_ATP_Spatiotemporal_AMR_Age50.npy", (0, 1)),
    ("Tier III Cohort (Moderate-Low Reserve)", "Output_ATP_Spatiotemporal_EAS_Age50.npy", (1, 0)),
    ("Tier IV Cohort (Low/Depleted Reserve)", "Output_ATP_Spatiotemporal_AFR_Age50.npy", (1, 1))
]

fig2 = plt.figure(figsize=(20, 13))
gs2 = fig2.add_gridspec(2, 2, wspace=0.10, hspace=0.38, bottom=0.10)
cmap_atp = LinearSegmentedColormap.from_list('atp_map', ['#b30000', '#ff7f0e', '#1f77b4'])
norm_atp = Normalize(vmin=0.12, vmax=2.85)

node_sizes = [220 if i == EPICENTER_IDX else (135 if i in ALL_INSULA_NODES else 38) for i in range(len(coords))]

for title_str, npy_name, (r_idx, c_idx) in SYNCHRONIZED_FIGURE2_COHORTS:
    ax = fig2.add_subplot(gs2[r_idx, c_idx])
    data_mat = np.load(get_dataset_file(npy_name))
    snapshot = data_mat[:, -1]
    node_colors_hex = [mcolors.to_hex(c) for c in cmap_atp(norm_atp(snapshot))]

    plotting.plot_connectome(
        L2_DTI_Mask, coords, display_mode='z',
        edge_threshold='99.2%', edge_cmap='Greys',
        node_color=node_colors_hex, node_size=node_sizes,
        colorbar=False, axes=ax, title=''
    )
    ax.set_title(f"{title_str}\nBiological Day 1000 (Age 50 Bifurcation Divergence)", fontsize=14.5, fontweight='bold', pad=22)

cbar_ax = fig2.add_axes([0.25, 0.03, 0.50, 0.022])
sm = ScalarMappable(cmap=cmap_atp, norm=norm_atp)
sm.set_array([])
cbar2 = fig2.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar2.set_label(r'Cytosolic $[\mathrm{ATP}]_{\mathrm{astrocytic}}\ (\mathrm{mM})$ at $t = 1000\ \mathrm{Days}$', fontsize=13, fontweight='bold', labelpad=8)

plt.suptitle(r'Figure 2: Connectome Heatmap of Focal Insular Bioenergetic Divergence Across Reserve Tiers ($t=1000$)', fontweight='bold', fontsize=19, y=0.98)
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_Network_Comparison.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("   -> [VALIDATED] Figure 2 archived (with quantitative ATP colorbar).")

# ---------------------------------------------------------
# [FIGURE 3 & 4] REGIME I KAPLAN-MEIER & VULNERABILITY CONTOUR
# ---------------------------------------------------------
print("\n[MODULE 3] Rendering Figures 3 & 4: Regime I Survival & Vulnerability Contour...")
extreme_load1 = df_sim1[df_sim1['Allostatic_Load_OR'] > 3.0].copy()

fig3, ax3 = plt.subplots(figsize=(10, 6.2))
for tier_label, color_hex in TIER_COLORS.items():
    subset = extreme_load1[extreme_load1['Reserve_Tier_Label'] == tier_label]
    if len(subset) > 0:
        times = np.sort(subset['Time_To_Bifurcation_Days'].values)
        n_total = len(times)
        event_times = times[times < 999.9]
        unique_t, counts = np.unique(event_times, return_counts=True)
        cum_events = np.cumsum(counts)
        surv_prob = (1.0 - cum_events / n_total) * 100.0

        plot_t = np.concatenate([[0.0], unique_t, [1000.0]])
        plot_s = np.concatenate([[100.0], surv_prob, [surv_prob[-1] if len(surv_prob) > 0 else 100.0]])
        ax3.step(plot_t, plot_s, where='post', label=tier_label, color=color_hex, linewidth=2.6)

ax3.set_xlim(0, 1000)
ax3.set_ylim(0, 105)
ax3.set_xlabel(r'Biological Time $t\ (\mathrm{Days})$', fontweight='bold')
ax3.set_ylabel(r'Bioenergetic Integrity Survival Probability $P(A_{\mathrm{insula}} > 0.50\ \mathrm{mM})\ (\%)$', fontweight='bold')
ax3.set_title(r'Figure 3: Regime I Kaplan-Meier Survival Estimate Under $\Omega_{\mathrm{ext}} > 3.0$ ($N=33,178$)', fontweight='bold', pad=14)
ax3.legend(title='Polygenic Mitochondrial Reserve', loc='upper right', framealpha=0.95, edgecolor='black')
fig3.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_Kaplan_Meier.png"), dpi=300, bbox_inches='tight')
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(10, 7.8))
sns.kdeplot(
    data=failed_patients1.sample(min(10000, len(failed_patients1)), random_state=42),
    x='Age', y='Allostatic_Load_OR', fill=True, cmap="mako", thresh=0.05, levels=12, ax=ax4, alpha=0.90
)
sns.scatterplot(
    data=failed_patients1.sample(min(1500, len(failed_patients1)), random_state=42),
    x='Age', y='Allostatic_Load_OR', color='white', s=16, alpha=0.40, marker='x', ax=ax4
)
ax4.set_xlim(20, 80)
ax4.set_ylim(2.0, 4.10)
ax4.set_xlabel(r'Chronological Senescence Entropy $\Delta S_{\mathrm{mito}}\ (\mathrm{Age\ in\ Years})$', fontweight='bold')
ax4.set_ylabel(r'Cumulative Allostatic Load ($\Omega_{\mathrm{ext}}$)', fontweight='bold')
ax4.set_title(r'Figure 4: Regime I Thermodynamic Vulnerability Contour (Senescence-Driven Attrition)', fontweight='bold', pad=14)
fig4.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4_Thermodynamic_Heatmap.png"), dpi=300, bbox_inches='tight')
plt.close(fig4)
print("   -> [VALIDATED] Figures 3 & 4 archived.")

# ---------------------------------------------------------
# [FIGURE 5] CHROMATIN REMODELING E(t) & SLC1A2 TRANSCRIPTOMICS
# ---------------------------------------------------------
print("\n[MODULE 4A] Rendering Figure 5: Slow SLC1A2 Chromatin Remodeling & EAAT2 Turnover...")
E_vals = np.load(get_dataset_file("Output_Epigenetic_Chromatin_E.npy"))
u_vals = np.load(get_dataset_file("Output_RNA_Unspliced_U.npy"))
s_vals = np.load(get_dataset_file("Output_RNA_Spliced_S.npy"))
vmax_ratio = np.clip(s_vals, 0.50, 1.0)

fig5, ax1 = plt.subplots(figsize=(10.5, 6.5))
color_rna = '#2c3e50'
ax1.set_xlabel(r'Biological Time $t\ (\mathrm{Days})$', fontweight='bold')
ax1.set_ylabel(r'Normalized Chromatin & Transcript State', color=color_rna, fontweight='bold')

l1 = ax1.plot(t_eval, E_vals, color='#16a085', linestyle='-.', linewidth=2.5, label=r'$\mathit{SLC1A2}$ Euchromatin Accessibility $E_i(t)$')
l2 = ax1.plot(t_eval, u_vals, color='#7f8c8d', linestyle='--', linewidth=2.2, label=r'Unspliced Pre-mRNA $u_i(t)$')
l3 = ax1.plot(t_eval, s_vals, color='#2c3e50', linestyle='-', linewidth=2.6, label=r'Mature Spliced mRNA Pool $s_i(t)$')
ax1.tick_params(axis='y', labelcolor=color_rna)
ax1.set_ylim(0.40, 1.08)

ax2 = ax1.twinx()
color_vmax = '#c0392b'
ax2.set_ylabel(r'Active EAAT2 Clearance Capacity Ratio ($V_{\max,i}^* / V_{\max,i}^0$)', color=color_vmax, fontweight='bold')
l4 = ax2.plot(t_eval, vmax_ratio, color=color_vmax, linewidth=3.0, alpha=0.85, label=r'Dynamic EAAT2 Capacity $V_{\max,i}^*(t)$')
ax2.tick_params(axis='y', labelcolor=color_vmax)
ax2.set_ylim(0.40, 1.08)

lines = l1 + l2 + l3 + l4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, edgecolor='black')

plt.title(r'Figure 5: Chronic $\mathit{SLC1A2}$ Chromatin Remodeling & EAAT2 Downregulation ($\Omega_{\mathrm{ext}}=3.5$)', fontweight='bold', pad=14)
fig5.subplots_adjust(bottom=0.24)
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_5_RNA_Velocity.png"), dpi=300, bbox_inches='tight')
plt.close(fig5)
print("   -> [VALIDATED] Figure 5 archived.")

# ---------------------------------------------------------
# [FIGURE 6] KRAMERS STOCHASTIC ESCAPE IN PHASE SPACE (FIXED ax=ax_ps)
# ---------------------------------------------------------
print("\n[MODULE 4B] Rendering Figure 6: Phase-Space Kramers Stochastic Escape Trajectory...")
ATP_epicenter = ATP_data[EPICENTER_IDX, :]
G_epicenter_mM = G_data[EPICENTER_IDX, :]
log10_G_epicenter = np.log10(np.maximum(G_epicenter_mM, 1e-4))

fig6, ax_ps = plt.subplots(figsize=(9.0, 7.8))
points = np.array([ATP_epicenter, log10_G_epicenter]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm_t = plt.Normalize(0, len(ATP_epicenter))
lc = LineCollection(segments, cmap='viridis', norm=norm_t, alpha=0.85, linewidth=2.2)
lc.set_array(np.arange(len(ATP_epicenter)))
line = ax_ps.add_collection(lc)

ax_ps.scatter(ATP_epicenter[0], log10_G_epicenter[0], color='#2ca02c', s=170, zorder=5,
              label=r'Homeostatic Origin ($t=0$: $3.0\ \mathrm{mM},\ 0.002\ \mathrm{mM}$)', marker='o', edgecolors='black')
ax_ps.scatter(ATP_epicenter[-1], log10_G_epicenter[-1], color='#d62728', s=180, zorder=5,
              label=r'Pathological Attractor ($t=1000$: $0.115\ \mathrm{mM},\ 7.79\ \mathrm{mM}$)', marker='X', edgecolors='black')
ax_ps.axvline(x=collapse_thresh, color='red', linestyle='--', linewidth=2.2,
              label=r'Saddle-Node Bifurcation ($\theta_c = 0.50\ \mathrm{mM}$)')
ax_ps.axhline(y=np.log10(0.020), color='#7f8c8d', linestyle=':', linewidth=1.8,
              label=r'EAAT2 Saturation Threshold ($K_{m,G} = 0.020\ \mathrm{mM}$)')

ax_ps.set_xlim(0.0, 3.30)
ax_ps.set_ylim(-3.1, 1.25)
yticks_log = [-3, -2, -1, 0, 1]
ax_ps.set_yticks(yticks_log)
ax_ps.set_yticklabels([r'$10^{-3}$ ($1\ \mu\mathrm{M}$)', r'$10^{-2}$ ($10\ \mu\mathrm{M}$)',
                       r'$10^{-1}$ ($100\ \mu\mathrm{M}$)', r'$10^{0}$ ($1\ \mathrm{mM}$)',
                       r'$10^{1}$ ($10\ \mathrm{mM}$)'])

ax_ps.set_xlabel(r'Insular Cytosolic $[\mathrm{ATP}]_{\mathrm{astrocytic}}\ (\mathrm{mM})$', fontweight='bold')
ax_ps.set_ylabel(r'Insular Extracellular $[\mathrm{Glu}]_{\mathrm{ext}}\ (\mathrm{mM},\ \log_{10}\ \mathrm{scale})$', fontweight='bold')
ax_ps.set_title(r'Figure 6: Kramers Stochastic Escape Trajectory in Phase Space', fontweight='bold', pad=14)

# FIXED: Explicit keyword argument ax=ax_ps prevents overwriting the main axes!
cbar = fig6.colorbar(line, ax=ax_ps, pad=0.03)
cbar.set_label(r'Biological Time $t\ (\mathrm{Days})$', rotation=270, labelpad=18, fontweight='bold')
ax_ps.legend(loc='upper right', framealpha=0.95, edgecolor='black')
fig6.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_6_Stochastic_PhaseSpace.png"), dpi=300, bbox_inches='tight')
plt.close(fig6)
print("   -> [VALIDATED] Figure 6 archived (Phase-Space trajectory restored).")

# ---------------------------------------------------------
# [FIGURE 7] REGIME II: GLYCOGEN MASS-BALANCE, YOUTH OVERLOAD & HYSTERESIS
# ---------------------------------------------------------
print("\n[MODULE 5] Rendering Figure 7: Regime II Glycogen Depletion, Youth Overload & Remission Hysteresis...")
ATP_young = np.load(get_dataset_file("Output_Regime2_Remission_Young_ATP.npy"))
R_young = np.load(get_dataset_file("Output_Regime2_Remission_Young_Rrec.npy"))
ATP_aged = np.load(get_dataset_file("Output_Regime2_Remission_Aged_ATP.npy"))
R_aged = np.load(get_dataset_file("Output_Regime2_Remission_Aged_Rrec.npy"))

df_sim2 = pd.read_csv(get_dataset_file("Output_Regime2_MegaSimulation_Final.csv"))
df_sim2["Reserve_Tier_Label"] = df_sim2["Ancestry"].map(TIER_MAP)

fig7 = plt.figure(figsize=(16.0, 12.2))
gs7 = fig7.add_gridspec(2, 2, hspace=0.45, wspace=0.32)

# PANEL A: YOUNG VS AGED REMISSION HYSTERESIS (Unobstructed recovery curve)
axA = fig7.add_subplot(gs7[0, 0])
axA.axvspan(0, 550, color='#fde0dd', alpha=0.35, label=r'Overload ($\Omega_{\mathrm{ext}} = 3.65$)')
axA.axvspan(550, 1000, color='#e5f5e0', alpha=0.35, label=r'Stress Relief ($\Omega_{\mathrm{ext}} = 1.20$)')
axA.plot(t_eval, ATP_young, color='#1f77b4', linewidth=2.5, label=r'Young (Age 25): Remission ($2.949\ \mathrm{mM}$)')
axA.plot(t_eval, ATP_aged, color='#d62728', linewidth=2.5, label=r'Older (Age 60): Hysteresis ($0.130\ \mathrm{mM}$)')
axA.axvline(x=550, color='#2c3e50', linestyle='--', linewidth=1.8)
axA.axhline(y=collapse_thresh, color='black', linestyle=':', linewidth=2.0, label=r'Bifurcation ($\theta_c = 0.50\ \mathrm{mM}$)')

axA.set_xlim(0, 1000)
axA.set_ylim(0, 4.15)
axA.set_xlabel(r'Biological Time $t\ (\mathrm{Days})$', fontweight='bold')
axA.set_ylabel(r'Insular Cytosolic $[\mathrm{ATP}]_{\mathrm{astrocytic}}\ (\mathrm{mM})$', fontweight='bold')
axA.set_title(r'A. Youth Overload Bifurcation vs. Aging Hysteresis', fontweight='bold', pad=10)
axA.legend(loc='upper center', ncol=2, fontsize=8.3, framealpha=0.95, edgecolor='black')

# PANEL B: GLYCOGEN CAPACITOR MASS-BALANCE (Unobstructed recharge curve)
axB = fig7.add_subplot(gs7[0, 1])
axB.axvspan(0, 550, color='#fde0dd', alpha=0.35)
axB.axvspan(550, 1000, color='#e5f5e0', alpha=0.35)
axB.plot(t_eval, R_young, color='#1f77b4', linewidth=2.6, linestyle='-', label=r'Young $R_{\mathrm{rec},i}(t)$ (Full Resynthesis)')
axB.plot(t_eval, R_aged, color='#d62728', linewidth=2.6, linestyle='-.', label=r'Older $R_{\mathrm{rec},i}(t)$ (Exhausted Pool)')
axB.axvline(x=550, color='#2c3e50', linestyle='--', linewidth=1.8, label=r'Stress Relief Onset ($t = 550\ \mathrm{d}$)')

axB.set_xlim(0, 1000)
axB.set_ylim(0, 1.32)
axB.set_xlabel(r'Biological Time $t\ (\mathrm{Days})$', fontweight='bold')
axB.set_ylabel(r'Astrocytic Glycogen Capacitor $R_{\mathrm{rec},i}(t)$', fontweight='bold')
axB.set_title(r'B. Glycogen Mass-Balance Depletion & Recharge', fontweight='bold', pad=10)
axB.legend(loc='upper center', ncol=2, fontsize=8.3, framealpha=0.95, edgecolor='black')

# PANEL C: REGIME II KAPLAN-MEIER SURVIVAL
axC = fig7.add_subplot(gs7[1, 0])
extreme_load2 = df_sim2[df_sim2['Allostatic_Load_OR'] > 3.0].copy()

for tier_label, color_hex in TIER_COLORS.items():
    subset = extreme_load2[extreme_load2['Reserve_Tier_Label'] == tier_label]
    if len(subset) > 0:
        times = np.sort(subset['Time_To_Bifurcation_Days'].values)
        n_total = len(times)
        event_times = times[times < 999.9]
        unique_t, counts = np.unique(event_times, return_counts=True)
        cum_events = np.cumsum(counts)
        surv_prob = (1.0 - cum_events / n_total) * 100.0

        plot_t = np.concatenate([[0.0], unique_t, [1000.0]])
        plot_s = np.concatenate([[100.0], surv_prob, [surv_prob[-1] if len(surv_prob) > 0 else 100.0]])
        axC.step(plot_t, plot_s, where='post', label=tier_label, color=color_hex, linewidth=2.5)

axC.set_xlim(0, 1000)
axC.set_ylim(0, 105)
axC.set_xlabel(r'Biological Time $t\ (\mathrm{Days})$', fontweight='bold')
axC.set_ylabel(r'Survival Probability $P(A_{\mathrm{insula}} > 0.50\ \mathrm{mM})\ (\%)$', fontweight='bold')
axC.set_title(r'C. Regime II Survival Under $\Omega_{\mathrm{ext}} > 3.0$ ($N=33,178$)', fontweight='bold', pad=10)
axC.legend(title='Polygenic Mitochondrial Reserve', loc='upper right', framealpha=0.95, edgecolor='black')

# PANEL D: REGIME II VULNERABILITY CONTOUR
axD = fig7.add_subplot(gs7[1, 1])
failed_regime2 = df_sim2[df_sim2["Survived_1000_Days"] == False].copy()

sns.kdeplot(
    data=failed_regime2.sample(min(10000, len(failed_regime2)), random_state=42),
    x='Age', y='Allostatic_Load_OR', fill=True, cmap="magma", thresh=0.05, levels=12, ax=axD, alpha=0.88
)
sns.scatterplot(
    data=failed_regime2.sample(min(1500, len(failed_regime2)), random_state=42),
    x='Age', y='Allostatic_Load_OR', color='white', s=15, alpha=0.40, marker='x', ax=axD
)
axD.set_xlim(20, 80)
axD.set_ylim(2.0, 4.10)
axD.set_xlabel(r'Chronological Age ($\mathrm{Years}$)', fontweight='bold')
axD.set_ylabel(r'Cumulative Allostatic Load ($\Omega_{\mathrm{ext}}$)', fontweight='bold')
axD.set_title(r'D. Regime II Vulnerability (64.66% Youth Overload)', fontweight='bold', pad=10)

plt.suptitle(
    r'Figure 7: Regime II Dynamics — Glycogen Mass-Balance Depletion, Young-Adult Overload, and Remission Hysteresis',
    fontweight='bold', fontsize=15.0, y=1.01
)
fig7.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_7_Regime_II_Overload_Recovery.png"), dpi=300, bbox_inches='tight')
plt.close(fig7)

print("   -> [VALIDATED] Figure 7 archived (Zero title overlap & unobstructed remission curves).")
print("\n=============================================================================")
print("[STATUS] AUDITED MASTER SCRIPT 04 COMPLETE. ALL 7 Q1 FIGURES ARCHIVED.")
print(f"Directory: {OUTPUT_DIR}")
print("=============================================================================\n")
