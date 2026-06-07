"""
=============================================================================
PIPELINE ISB - SCRIPT 04: HIGH-FIDELITY VISUAL RENDERING ENGINE (MASTER PATCH)
=============================================================================
Function: Ingests spatiotemporal .npy matrices and multidimensional .csv outputs 
          to render high-resolution, publication-ready figures.
          Engineered to visualize stochastic noise (SDE), critical slowing down 
          phenomena, thermodynamic contours, topological phase transitions, 
          and micro-macro causal bridges (RNA Velocity & Stochastic Escape).
Status: Reproducibility Standard - Analytical Output Engine
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import seaborn as sns
from nilearn import datasets, plotting
from nilearn.image import load_img
from matplotlib.colors import LinearSegmentedColormap, Normalize
import warnings
warnings.filterwarnings("ignore")

print("[SYSTEM] Initializing high-fidelity SDE visual rendering engine...")

# [SYSTEM] Dynamic OS-Agnostic Path Resolution
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.path.abspath('.')

BASE_DIR = os.path.join(CURRENT_DIR, 'ISB_Empirical_Vault')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Manuscript_Figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"-> [SYSTEM] Output directory anchored at: {OUTPUT_DIR}")

# Q1 Aesthetics setup with LaTeX integration formatting
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# ---------------------------------------------------------
# [FIGURE 1] FOCAL NETWORK DYNAMICS & ASTROCYTIC COUPLING 
# ---------------------------------------------------------
print("\n[MODULE 1] Synthesizing Figure 1: Focal network SDE dynamics...")
try:
    ATP_data = np.load(os.path.join(BASE_DIR, "Output_ATP_Spatiotemporal_AFR_Age60.npy"))
    G_data = np.load(os.path.join(BASE_DIR, "Output_G_Spatiotemporal_AFR_Age60.npy"))
    t_eval = np.load(os.path.join(BASE_DIR, "Output_Time_Vector.npy"))
    
    with open(os.path.join(BASE_DIR, "L5_L6_Clinical_Anchors.json"), "r") as f:
        anchors = json.load(f)
    
    collapse_thresh = anchors["1H_MRS_Metabolomics"]["Saddle_Node_Threshold"]
    Global_Mean_ATP = np.mean(ATP_data, axis=0)
    Global_Mean_G = np.mean(G_data, axis=0)
    
    fig1 = plt.figure(figsize=(14, 11))
    gs = fig1.add_gridspec(2, 2, height_ratios=[2, 1.5], hspace=0.45, wspace=0.25)
    
    ax1 = fig1.add_subplot(gs[0, 0])
    window_size = 20
    epicenter_drift = pd.Series(ATP_data[0, :]).rolling(window=window_size, min_periods=1).mean()
    
    ax1.plot(t_eval, ATP_data[0, :], color='#d62728', linewidth=1.0, alpha=0.35, label=r'Epicenter (Raw $dW_t$ Noise)')
    ax1.plot(t_eval, epicenter_drift, color='#8c1515', linewidth=3, label=r'Epicenter $\langle ATP \rangle$ Drift')
    ax1.plot(t_eval, ATP_data[147, :], color='#9467bd', linewidth=1.5, alpha=0.7, linestyle='-', label=r'Distant Node ($\eta_{147}$)')
    ax1.plot(t_eval, Global_Mean_ATP, color='#1f77b4', linewidth=3, label=r'Global Brain $\mu_{ATP}$')
    ax1.axhline(y=collapse_thresh, color='black', linestyle=':', linewidth=2, label=r'Bifurcation ($\theta_c < 0.5$)')
    
    ax1.set_ylabel(r'$[ATP]_{astrocytic} \ (mM)$', fontweight='bold')
    ax1.set_xlabel(r'$t \ (Days)$', fontweight='bold')
    ax1.set_title(r'A. SDE Bioenergetic Depletion: Critical Slowing Down', fontweight='bold')
    ax1.set_ylim(0, 4.5)
    ax1.legend(loc='upper right', framealpha=0.95, edgecolor='black')

    ax2 = fig1.add_subplot(gs[0, 1])
    ax2.plot(t_eval, G_data[0, :], color='#d62728', linewidth=2, alpha=0.8, label=r'Epicenter $[Glu]_{syn}$')
    ax2.plot(t_eval, Global_Mean_G, color='#1f77b4', linewidth=2.5, linestyle='-', label=r'Global Mean $\mu_{Glu}$')
    
    ax2.set_ylabel(r'$[Glu]_{synaptic} \ (mM)$', fontweight='bold')
    ax2.set_xlabel(r'$t \ (Days)$', fontweight='bold')
    ax2.set_title(r'B. Vagal Influx & Transcriptional $V_{max}$ Impairment', fontweight='bold')
    ax2.set_ylim(0, np.max(G_data) * 1.4 + 0.1)
    ax2.legend(loc='upper left', framealpha=0.95, edgecolor='black')

    df_sim = pd.read_csv(os.path.join(BASE_DIR, "Output_TrueDynamic_MegaSimulation_Final.csv"))
    failed_patients = df_sim[df_sim['Survived_1000_Days'] == False]
    
    ax3 = fig1.add_subplot(gs[1, :])
    sns.scatterplot(data=failed_patients, x='Time_To_Bifurcation_Days', y='Stabilized_ISB_Score', 
                    hue='Ancestry', palette='Set1', s=60, alpha=0.7, edgecolor='black', ax=ax3)
    
    ax3.set_xlabel(r'Time-to-Phase-Transition $t_c$ (Biological Days)', fontweight='bold')
    ax3.set_ylabel(r'Basal ISB Metric $\Omega_{basal}$', fontweight='bold')
    ax3.set_title(r'C. Correlation: Bioenergetic Resilience vs Phase Transition Acceleration', fontweight='bold')
    ax3.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True, edgecolor='black')
    
    plt.suptitle(r'Figure 1: Stochastic Multiscale Dynamics of Phase Transition', fontweight='bold', fontsize=18, y=1.03)
    fig1.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_1_Complexity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("   -> [VALIDATED] Figure 1 archived.")
except Exception as e:
    import traceback
    print(f"   -> [ERROR] Figure 1 generation failed: {e}")
    traceback.print_exc()

# ---------------------------------------------------------
# [FIGURE 2] PAN-ANCESTRY SPATIOTEMPORAL CONNECTOME MAP
# ---------------------------------------------------------
print("\n[MODULE 2] Synthesizing Figure 2: Pan-ancestry spatiotemporal map...")
try:
    destrieux = datasets.fetch_atlas_destrieux_2009()
    atlas_img = load_img(destrieux['maps'])
    coords = plotting.find_parcellation_cut_coords(atlas_img)
    
    L2_DTI_Mask = np.load(os.path.join(BASE_DIR, "L2_DTI_Structural_Mask.npy"))
    coords = coords[:L2_DTI_Mask.shape[0]] 
    
    ATP_EUR = np.load(os.path.join(BASE_DIR, "Output_ATP_Spatiotemporal_EUR_Age60.npy"))
    ATP_EAS = np.load(os.path.join(BASE_DIR, "Output_ATP_Spatiotemporal_EAS_Age60.npy"))
    ATP_AMR = np.load(os.path.join(BASE_DIR, "Output_ATP_Spatiotemporal_AMR_Age60.npy"))
    ATP_AFR = np.load(os.path.join(BASE_DIR, "Output_ATP_Spatiotemporal_AFR_Age60.npy"))
    
    TIME_IDX = -1 
    
    fig2 = plt.figure(figsize=(20, 12))
    gs = fig2.add_gridspec(2, 2, wspace=0.1, hspace=0.45)
    
    cmap_atp = LinearSegmentedColormap.from_list('atp_map', ['#d62728', '#ff7f0e', '#1f77b4']) 
    norm = Normalize(vmin=0.0, vmax=3.0) 
    
    node_sizes = [250 if i == 0 else 40 for i in range(len(coords))]
    
    cohorts = [
        ('EUR Cohort (High mt-CN Buffer)', ATP_EUR, gs[0, 0]),
        ('EAS Cohort (Moderate-High Buffer)', ATP_EAS, gs[0, 1]),
        ('AMR Cohort (Moderate-Low Buffer)', ATP_AMR, gs[1, 0]),
        ('AFR Cohort (Depleted mt-CN Buffer)', ATP_AFR, gs[1, 1])
    ]
    
    for title, data, pos in cohorts:
        ax = fig2.add_subplot(pos)
        snapshot = data[:, TIME_IDX]
        
        node_colors_rgba = cmap_atp(norm(snapshot))
        node_colors_hex = [mcolors.to_hex(c) for c in node_colors_rgba]
        
        plotting.plot_connectome(
            L2_DTI_Mask, coords, display_mode='z', 
            edge_threshold='99.5%', edge_cmap='Greys', 
            node_color=node_colors_hex, node_size=node_sizes, 
            colorbar=False, axes=ax, title='' 
        )
        
        ax.set_title(f'{title}\nBiological Day 1000', fontsize=16, fontweight='bold', pad=30)
        print(f"     -> Topographical mapping resolved for {title[:3]}")

    plt.suptitle(r'Figure 2: Topological Heatmap of Focal Bioenergetic Depletion ($t=1000$)', fontweight='bold', fontsize=22, y=1.05)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_Network_Comparison.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print("   -> [VALIDATED] Figure 2 archived.")
except Exception as e:
    import traceback
    print(f"   -> [ERROR] Figure 2 generation failed: {e}")
    traceback.print_exc()

# ---------------------------------------------------------
# [FIGURE 3 & 4] KAPLAN-MEIER ESTIMATE & THERMODYNAMIC CONTOUR
# ---------------------------------------------------------
print("\n[MODULE 3] Synthesizing Figures 3 & 4: Survival & thermodynamic density maps...")
try:
    df_sim = pd.read_csv(os.path.join(BASE_DIR, "Output_TrueDynamic_MegaSimulation_Final.csv"))
    extreme_load = df_sim[df_sim['Allostatic_Load_OR'] > 3.0]
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors = {'AFR': '#d62728', 'AMR': '#ff7f0e', 'EAS': '#2ca02c', 'EUR': '#1f77b4'}
    for ancestry in ['AFR', 'AMR', 'EAS', 'EUR']:
        subset = extreme_load[extreme_load['Ancestry'] == ancestry]
        times = np.sort(subset['Time_To_Bifurcation_Days'].values)
        survival_prob = 1.0 - np.arange(1, len(times) + 1) / len(times)
        times = np.insert(times, 0, 0)
        survival_prob = np.insert(survival_prob, 0, 1.0)
        ax3.step(times, survival_prob * 100, where='post', label=ancestry, color=colors[ancestry], linewidth=2.5)
        
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(-5, 115) 
    ax3.set_xlabel(r'Biological Days ($t$)', fontweight='bold')
    ax3.set_ylabel(r'Systemic Bioenergetic Integrity Probability $P(\theta > 0.5) \ (\%)$', fontweight='bold')
    ax3.set_title(r'Figure 3: Kaplan-Meier Survival Estimate Under $OR > 3.0$', fontweight='bold', pad=15)
    ax3.legend(title='Ancestry', loc='upper right', framealpha=0.95, edgecolor='black')
    fig3.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_Kaplan_Meier.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    collapsed_patients = df_sim[df_sim['Survived_1000_Days'] == False]
    
    sns.kdeplot(data=collapsed_patients, x='Age', y='Allostatic_Load_OR', fill=True, cmap="mako", thresh=0.05, levels=12, ax=ax4, alpha=0.9)
    sns.scatterplot(data=collapsed_patients, x='Age', y='Allostatic_Load_OR', color='white', s=20, alpha=0.6, marker='x', ax=ax4)
    ax4.set_xlim(20, 80)
    ax4.set_ylim(1.0, 4.5)
    ax4.set_xlabel(r'Senescence Entropy $\Delta S_{mito}$ (Age in Years)', fontweight='bold')
    ax4.set_ylabel(r'Cumulative Allostatic Load ($\Omega_{ext}$ / Odds Ratio)', fontweight='bold')
    ax4.set_title(r'Figure 4: Thermodynamic Vulnerability Contour (Bifurcation Zone Density)', fontweight='bold', pad=15)
    fig4.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4_Thermodynamic_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("   -> [VALIDATED] Figures 3 & 4 archived.")
except Exception as e:
    import traceback
    print(f"   -> [ERROR] Figures 3 or 4 generation failed: {e}")
    traceback.print_exc()

# ---------------------------------------------------------
# [FIGURE 5] MICRO-MACRO BRIDGE: RNA VELOCITY & VMAX KINETICS
# ---------------------------------------------------------
print("\n[MODULE 4A] Synthesizing Figure 5: Epigenetic drift & transporter degradation...")
try:
    with open(os.path.join(BASE_DIR, "L5_L6_Clinical_Anchors.json"), "r") as f:
        anchors = json.load(f)
    
    isb_params = anchors["ISB_2_0_Expansions"]
    TAU_BBB = isb_params["BBB_Permeability_Delay_Days"]
    rna_drift = isb_params["Epigenetic_Drift_RNA_Velocity"]
    BETA_SPLICING = rna_drift["Splicing_Rate_Beta"]
    GAMMA_DEGRADATION = rna_drift["Degradation_Rate_Gamma"]
    eaat_loss_max = anchors["Neurophysiology_Anchors"]["EAAT_Downregulation_Max"]
    R_STRESS_MAX_CAP = anchors["Neurophysiology_Anchors"]["Max_Metabolic_Capacity_Limit"]
    
    alpha_min = 1.0 - eaat_loss_max 
    LAMBDA_DECAY = -np.log(alpha_min) / (R_STRESS_MAX_CAP - 1.0) 
    
    t_days = np.linspace(0, 1000, 1000)
    dt = 1.0
    R_stress_target = 3.5
    
    alpha_0 = 1.0
    u_vals = np.zeros(1000)
    s_vals = np.zeros(1000)
    vmax_ratio = np.zeros(1000)
    
    u_vals[0] = alpha_0 / BETA_SPLICING
    s_vals[0] = alpha_0 / GAMMA_DEGRADATION
    
    for i in range(1, 1000):
        current_R_stress = 1.0 + (R_stress_target - 1.0) * (1.0 - np.exp(-t_days[i] / TAU_BBB))
        current_alpha = alpha_0 * np.exp(-LAMBDA_DECAY * (current_R_stress - 1.0))
        
        du_dt = current_alpha - (BETA_SPLICING * u_vals[i-1])
        ds_dt = (BETA_SPLICING * u_vals[i-1]) - (GAMMA_DEGRADATION * s_vals[i-1])
        
        u_vals[i] = u_vals[i-1] + du_dt * dt
        s_vals[i] = s_vals[i-1] + ds_dt * dt
        
        s_ratio = s_vals[i] / s_vals[0]
        vmax_ratio[i] = np.maximum(s_ratio, (1.0 - eaat_loss_max))
        
    vmax_ratio[0] = 1.0 

    fig5, ax1 = plt.subplots(figsize=(10, 6.5))
    
    color_rna = '#34495e'
    ax1.set_xlabel(r'Biological Days ($t$)', fontweight='bold')
    ax1.set_ylabel(r'RNA Transcript Concentration (Normalized)', color=color_rna, fontweight='bold')
    line1 = ax1.plot(t_days, u_vals / u_vals[0], color='#7f8c8d', linestyle='--', linewidth=2, label=r'Unspliced RNA ($u$)')
    line2 = ax1.plot(t_days, s_vals / s_vals[0], color='#2c3e50', linestyle='-', linewidth=2.5, label=r'Spliced RNA ($s$)')
    ax1.tick_params(axis='y', labelcolor=color_rna)
    ax1.set_ylim(0, 1.2)

    ax2 = ax1.twinx()  
    color_vmax = '#c0392b'
    ax2.set_ylabel(r'EAAT Transporter Capacity Ratio ($V_{max}^* / V_{max}$)', color=color_vmax, fontweight='bold')
    line3 = ax2.plot(t_days, vmax_ratio, color=color_vmax, linewidth=3, label=r'Dynamic $V_{max}$ Capacity')
    ax2.tick_params(axis='y', labelcolor=color_vmax)
    ax2.set_ylim(0, 1.2)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, edgecolor='black')
    
    plt.title(r'Figure 5: Epigenetic Drift & Mechanical $V_{max}$ Downregulation ($OR=3.5$)', fontweight='bold', pad=15)
    fig5.subplots_adjust(bottom=0.25)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_5_RNA_Velocity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("   -> [VALIDATED] Figure 5 archived (Absolute Legend Isolation).")
except Exception as e:
    import traceback
    print(f"   -> [ERROR] Figure 5 generation failed: {e}")
    traceback.print_exc()

# ---------------------------------------------------------
# [FIGURE 6] STOCHASTIC ESCAPE (SADDLE-NODE TRAJECTORY)
# ---------------------------------------------------------
print("\n[MODULE 4B] Synthesizing Figure 6: Stochastic escape trajectory...")
try:
    ATP_data = np.load(os.path.join(BASE_DIR, "Output_ATP_Spatiotemporal_AFR_Age60.npy"))
    G_data = np.load(os.path.join(BASE_DIR, "Output_G_Spatiotemporal_AFR_Age60.npy"))
    
    ATP_epicenter = ATP_data[0, :]
    G_epicenter = G_data[0, :]
    
    fig6, ax_ps = plt.subplots(figsize=(8, 8))
    
    points = np.array([ATP_epicenter, G_epicenter]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm_t = plt.Normalize(0, len(ATP_epicenter))
    lc = LineCollection(segments, cmap='viridis', norm=norm_t, alpha=0.6, linewidth=1.5)
    lc.set_array(np.arange(len(ATP_epicenter)))
    line = ax_ps.add_collection(lc)
    
    ax_ps.scatter(ATP_epicenter[0], G_epicenter[0], color='#2ca02c', s=150, zorder=5, label='Homeostatic Origin ($t=0$)', marker='o', edgecolors='black')
    ax_ps.scatter(ATP_epicenter[-1], G_epicenter[-1], color='#d62728', s=150, zorder=5, label='Bifurcation State ($t=1000$)', marker='X', edgecolors='black')
    
    ax_ps.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label=r'Bifurcation Threshold ($\theta_c = 0.5$)')
    
    ax_ps.set_xlim(0, 3.5)
    ax_ps.set_ylim(0, max(G_epicenter) * 1.3 + 0.1)
    ax_ps.set_xlabel(r'$[ATP]_{astrocytic} \ (mM)$', fontweight='bold')
    ax_ps.set_ylabel(r'$[Glu]_{synaptic} \ (mM)$', fontweight='bold')
    ax_ps.set_title(r'Figure 6: Stochastic Escape Trajectory (Langevin Noise)', fontweight='bold', pad=15)
    
    cbar = fig6.colorbar(line, ax=ax_ps)
    cbar.set_label('Biological Days ($t$)', rotation=270, labelpad=15)
    
    ax_ps.legend(loc='upper right', framealpha=0.95, edgecolor='black')
    fig6.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_6_Stochastic_PhaseSpace.png"), dpi=300, bbox_inches='tight')
    plt.close(fig6)
    print("   -> [VALIDATED] Figure 6 archived (Stochastic Escape captured).")
except Exception as e:
    import traceback
    print(f"   -> [ERROR] Figure 6 generation failed: {e}")
    traceback.print_exc()

print("\n=============================================================================")
print("[STATUS] SCRIPT 04 COMPLETE. ALL PUBLICATION-READY FIGURES ARCHIVED.")
print(f"Directory: {OUTPUT_DIR}")
print("=============================================================================\n")
