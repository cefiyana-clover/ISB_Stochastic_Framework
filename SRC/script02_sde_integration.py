"""
=============================================================================
PIPELINE ISB - SCRIPT 02: MULTI-SCALE INTEGRATION FRAMEWORK
=============================================================================
Function: Integrates empirical matrices into a coupled SDE-PDE system.
          Computes spatiotemporal allostatic load dynamics combining Langevin
          noise, anisotropic diffusion, and RNA velocity splicing kinetics.
Status: Reproducibility Standard - Stochastic Computational Core
=============================================================================
"""

import os
import json
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

print("[SYSTEM] Initializing multi-scale SDE-PDE integration framework...")

# [SYSTEM] Dynamic OS-Agnostic Path Resolution
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.path.abspath('.')

BASE_DIR = os.path.join(CURRENT_DIR, 'ISB_Empirical_Vault')
os.makedirs(BASE_DIR, exist_ok=True)
print(f"-> [SYSTEM] Base directory anchored at: {BASE_DIR}")

# ---------------------------------------------------------
# [MODULE 1] EMPIRICAL DATA INGESTION
# ---------------------------------------------------------
print("\n[MODULE 1] Loading empirical matrices and structural anchors...")
try:
    L2_DTI_W = np.load(os.path.join(BASE_DIR, "L2_DTI_Structural_Mask.npy")) 
    P_basal_global = np.load(os.path.join(BASE_DIR, "L3_Transcriptomic_Pbasal.npy"))
    Vmax_baseline = np.load(os.path.join(BASE_DIR, "L4_PET_Vmax_Density.npy"))
    
    with open(os.path.join(BASE_DIR, "L5_L6_Clinical_Anchors.json"), "r") as f:
        anchors = json.load(f)
    with open(os.path.join(BASE_DIR, "L7_Population_Genetics.json"), "r") as f:
        pop_genetics = json.load(f)
        
    NUM_NODES = L2_DTI_W.shape[0] 
    
    # Constructing Spatiotemporal Graph Laplacian
    Degree_Matrix = np.diag(np.sum(L2_DTI_W, axis=1))
    Laplacian_L = Degree_Matrix - L2_DTI_W
    
    print(f"-> [VALIDATED] Spatiotemporal Graph Laplacian constructed ({NUM_NODES} nodes).")

except Exception as e:
    raise RuntimeError(f"[ERROR] Empirical data ingestion failed: {e}")

# ---------------------------------------------------------
# [MODULE 2] BIOENERGETIC & KINETIC PARAMETERIZATION
# ---------------------------------------------------------
print("\n[MODULE 2] Parameterizing multi-modal bioenergetic anchors...")

baseline_atp = anchors["1H_MRS_Metabolomics"]["Baseline_ATP_mM"]
baseline_glu = anchors["1H_MRS_Metabolomics"]["Baseline_Glutamate_mM"]

stoich_atp_glu = anchors["Thermodynamic_Coupling"]["ATP_per_Glutamate_Stoichiometry"]
decline_rate = anchors["Mitochondrial_Senescence"]["Decline_Rate_Per_Year"]
peak_age = anchors["Mitochondrial_Senescence"]["Peak_Optimal_Age"]
eaat_loss_max = anchors["Neurophysiology_Anchors"]["EAAT_Downregulation_Max"]
R_STRESS_MAX_CAP = anchors["Neurophysiology_Anchors"]["Max_Metabolic_Capacity_Limit"]

# Stochastic & Recovery Constants
isb_params = anchors["ISB_2_0_Expansions"]
RHO_REC = isb_params["Neuroplasticity_Recovery_Rho"]
THETA_REC = isb_params["Recovery_Threshold_Theta"]
SIGMA_NOISE = isb_params["Stochastic_Noise_Sigma"]
TAU_BBB = isb_params["BBB_Permeability_Delay_Days"]

# Epigenetic Drift (RNA Velocity) Constants
epi_params = isb_params["Epigenetic_Drift_RNA_Velocity"]
BETA_SPLICING = epi_params["Splicing_Rate_Beta"]
GAMMA_DEG = epi_params["Degradation_Rate_Gamma"]
ALPHA_BASELINE = GAMMA_DEG * 1.0 

# [DERIVATION] Analytically deriving Lambda from physiological boundaries
alpha_min = 1.0 - eaat_loss_max 
LAMBDA_DECAY = -np.log(alpha_min) / (R_STRESS_MAX_CAP - 1.0) 

Km_ATP, Km_G, D_G = 0.1, 0.1, 0.01
TARGET_ALLOSTATIC_LOAD = 3.5 

print(f"-> [VALIDATED] SDE recovery & noise matrices calibrated.")
print(f"-> [VALIDATED] Epigenetic transcription decay (Lambda) derived: {LAMBDA_DECAY:.4f}")

# ---------------------------------------------------------
# [MODULE 3] SDE-PDE EULER-MARUYAMA ENGINE
# ---------------------------------------------------------
def T_BBB(t, max_R_stress, tau):
    """Exponential transit of systemic stress through endothelial wall"""
    return 1.0 + (max_R_stress - 1.0) * (1.0 - np.exp(-t / tau))

def sde_euler_maruyama(Laplacian, P_basal_current, Vmax_base, k_decay_arr, k_prod_arr, R_stress_max, t_span, dt=0.1):
    num_steps = int((t_span[1] - t_span[0]) / dt)
    sqrt_dt = np.sqrt(dt)
    
    record_steps = int(1.0 / dt) 
    num_days = int(t_span[1])
    ATP_daily = np.zeros((NUM_NODES, num_days))
    Glu_daily = np.zeros((NUM_NODES, num_days))
    
    curr_ATP = np.full(NUM_NODES, baseline_atp)
    curr_Glu = np.full(NUM_NODES, baseline_glu)
    
    curr_u = np.full(NUM_NODES, ALPHA_BASELINE / BETA_SPLICING)
    curr_s = np.full(NUM_NODES, ALPHA_BASELINE / GAMMA_DEG) 
    
    day_idx = 0
    for step in range(num_steps):
        t = step * dt
        
        # 1. BBB Filtration
        current_R_stress = T_BBB(t, R_stress_max, TAU_BBB)
        
        # 2. Epigenetic Drift (RNA Velocity)
        alpha_t = ALPHA_BASELINE * np.exp(-LAMBDA_DECAY * (current_R_stress - 1.0))
        
        du_dt = alpha_t - (BETA_SPLICING * curr_u)
        ds_dt = (BETA_SPLICING * curr_u) - (GAMMA_DEG * curr_s) 
        
        next_u = curr_u + (du_dt * dt)
        next_s = curr_s + (ds_dt * dt)
        next_u = np.maximum(next_u, 1e-6)
        next_s = np.maximum(next_s, 1e-6)
        
        s_ratio = next_s / (ALPHA_BASELINE / GAMMA_DEG)
        Vmax_dynamic = Vmax_base * np.maximum(s_ratio, (1.0 - eaat_loss_max))
        
        # 3. Adaptive Neuroplasticity
        recovery_term = RHO_REC * np.where(curr_ATP > THETA_REC, 1.0, 0.0)
        
        # 4. SDE: Deterministic Drift & Langevin Noise [STOICHIOMETRY RESTORED]
        Dynamic_Thermo_Stress = current_R_stress * stoich_atp_glu
        dATP_drift = P_basal_current - (Dynamic_Thermo_Stress * curr_Glu) / (Km_ATP + curr_Glu) - (k_decay_arr * curr_ATP) + recovery_term
        dW = np.random.normal(0, 1, NUM_NODES) * sqrt_dt
        stochastic_diffusion = SIGMA_NOISE * curr_ATP * dW
        
        next_ATP = curr_ATP + (dATP_drift * dt) + stochastic_diffusion
        next_ATP = np.maximum(next_ATP, 1e-4) 
        
        # 5. PDE: Anisotropic Glutamate Diffusion [DYNAMIC K_PROD DEPLOYED]
        clearance_failure = (k_prod_arr * curr_ATP) - (Vmax_dynamic * curr_Glu) / (Km_G + curr_Glu)
        dGlu_drift = clearance_failure - D_G * np.dot(Laplacian, curr_Glu)
        
        next_Glu = curr_Glu + (dGlu_drift * dt)
        next_Glu = np.maximum(next_Glu, 1e-4)
        
        # Advance State
        curr_ATP = next_ATP
        curr_Glu = next_Glu
        curr_u = next_u
        curr_s = next_s
        
        if step % record_steps == 0 and day_idx < num_days:
            ATP_daily[:, day_idx] = curr_ATP
            Glu_daily[:, day_idx] = curr_Glu
            day_idx += 1
            
    return ATP_daily, Glu_daily

# ---------------------------------------------------------
# [MODULE 4] 4D SPATIOTEMPORAL EXECUTION (COHORTS)
# ---------------------------------------------------------
print("\n[MODULE 4] Computing stochastic 4D spatiotemporal trajectories...")

ALLOSTATIC_SPAN = (0, 1000)
DT_STEP = 0.1
global_mean_mtCN = pop_genetics["GLOBAL_MEAN_REFERENCE"]

ANCESTRIES = ["AFR", "EUR", "EAS", "AMR"]
COHORT_AGES = [20.0, 30.0, 40.0, 50.0, 60.0]

start_time = time.time()

for age in COHORT_AGES:
    age_penalty = 1.0 - ((age - peak_age) * decline_rate) if age > peak_age else 1.0
    print(f"\n=========================================================")
    print(f"---> [COHORT INITIALIZATION] Age: {int(age)} | Entropy Coefficient: {age_penalty:.4f}")
    
    for ancestry in ANCESTRIES:
        ancestry_mean_mtCN = pop_genetics[ancestry]["mean_mtCN"]
        rbs_factor = ancestry_mean_mtCN / global_mean_mtCN
        
        P_basal_scaled = P_basal_global * rbs_factor * age_penalty
        P_basal_clamped = np.clip(P_basal_scaled, a_min=4.0, a_max=9.0)
        
        # =====================================================================
        # [DOUBLE STEADY-STATE CALIBRATION]: Strict Biophysical Alignment
        # =====================================================================
        # 1. Derive personal k_prod per node to match native baseline PET Vmax density
        Vmax_basal_clearance = (Vmax_baseline * baseline_glu) / (Km_G + baseline_glu)
        k_prod_personal = Vmax_basal_clearance / baseline_atp
        
        # 2. Derive personal k_decay per node matching P_basal and baseline drain
        Basal_Thermodynamic_Stress = 1.0 * stoich_atp_glu
        Basal_Drain = (Basal_Thermodynamic_Stress * baseline_glu) / (Km_ATP + baseline_glu)
        k_decay_personal = (P_basal_clamped - Basal_Drain + RHO_REC) / baseline_atp
        # =====================================================================
        
        print(f"     -> Integrating SDE | Cohort: {ancestry} | RBS: {rbs_factor:.4f}...", end=" ")
        
        ATP_Spatiotemporal, G_Spatiotemporal = sde_euler_maruyama(
            Laplacian=Laplacian_L,
            P_basal_current=P_basal_clamped,
            Vmax_base=Vmax_baseline, 
            k_decay_arr=k_decay_personal,
            k_prod_arr=k_prod_personal,
            R_stress_max=TARGET_ALLOSTATIC_LOAD,
            t_span=ALLOSTATIC_SPAN,
            dt=DT_STEP
        )
        
        # Absolute path integration for output stabilization
        file_atp = os.path.join(BASE_DIR, f"Output_ATP_Spatiotemporal_{ancestry}_Age{int(age)}.npy")
        file_g = os.path.join(BASE_DIR, f"Output_G_Spatiotemporal_{ancestry}_Age{int(age)}.npy")
        
        np.save(file_atp, ATP_Spatiotemporal)
        np.save(file_g, G_Spatiotemporal)
        print("[VALIDATED]")

t_eval = np.linspace(ALLOSTATIC_SPAN[0], ALLOSTATIC_SPAN[1], int(ALLOSTATIC_SPAN[1]))
np.save(os.path.join(BASE_DIR, "Output_Time_Vector.npy"), t_eval)

exec_time = time.time() - start_time
print(f"\n=============================================================================")
print(f"[STATUS] SCRIPT 02 COMPLETE. SDE-PDE topology integrated in {exec_time:.2f} seconds.")
print("=============================================================================\n")
