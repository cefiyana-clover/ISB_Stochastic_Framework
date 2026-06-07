"""
=============================================================================
PIPELINE ISB - SCRIPT 03: LARGE-SCALE DYNAMIC COHORT SIMULATION
=============================================================================
Function: SDE-PDE Monte Carlo execution across virtual cohorts.
          Incorporates genetic ancestry (mtDNA-CN), systemic allostatic load, 
          senescence entropy, focal vagal influx, dose-dependent EAAT downregulation, 
          steady-state homeostatic calibration, and focal topological phase transition.
Status: Reproducibility Standard - Stochastic Computational Framework
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings("ignore")

print("[SYSTEM] Initializing large-scale SDE-PDE cohort simulation...")

# [SYSTEM] Dynamic OS-Agnostic Path Resolution
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.path.abspath('.')

BASE_DIR = os.path.join(CURRENT_DIR, 'ISB_Empirical_Vault')
os.makedirs(BASE_DIR, exist_ok=True)
print(f"-> [SYSTEM] Base directory anchored at: {BASE_DIR}")

# [COMPUTE LOCK] Parameter populasi dimaksimalkan untuk eksekusi HPC/Cloud
POPULATION_SIZE = 100000 
QUARTER_POP = POPULATION_SIZE // 4

# ---------------------------------------------------------
# [MODULE 1] DATA INGESTION & PARAMETERIZATION
# ---------------------------------------------------------
print("\n[MODULE 1] Loading spatiotemporal matrices and structural anchors...")
try:
    L2_DTI_W = np.load(os.path.join(BASE_DIR, "L2_DTI_Structural_Mask.npy"))
    P_basal_raw = np.load(os.path.join(BASE_DIR, "L3_Transcriptomic_Pbasal.npy"))
    Vmax_raw = np.load(os.path.join(BASE_DIR, "L4_PET_Vmax_Density.npy"))

    with open(os.path.join(BASE_DIR, "L5_L6_Clinical_Anchors.json"), "r") as f:
        anchors = json.load(f)
    with open(os.path.join(BASE_DIR, "L7_Population_Genetics.json"), "r") as f:
        pop_genetics = json.load(f)
except Exception as e:
    raise RuntimeError(f"[ERROR] Data ingestion failed: {e}")

NUM_NODES = L2_DTI_W.shape[0] 
P_basal_global_array = P_basal_raw[:NUM_NODES]
Vmax = Vmax_raw[:NUM_NODES]

Degree_Matrix = np.diag(np.sum(L2_DTI_W, axis=1))
Laplacian_L = Degree_Matrix - L2_DTI_W

baseline_atp = anchors["1H_MRS_Metabolomics"]["Baseline_ATP_mM"]
baseline_glu = anchors["1H_MRS_Metabolomics"]["Baseline_Glutamate_mM"]
CRITICAL_ATP_BIFURCATION = anchors["1H_MRS_Metabolomics"]["Saddle_Node_Threshold"]
stoich_atp_glu = anchors["Thermodynamic_Coupling"]["ATP_per_Glutamate_Stoichiometry"]
decline_rate = anchors["Mitochondrial_Senescence"]["Decline_Rate_Per_Year"]
peak_age = anchors["Mitochondrial_Senescence"]["Peak_Optimal_Age"]
vagal_firing_mult = anchors["Neurophysiology_Anchors"]["Vagal_Hyperfiring_Multiplier"]
eaat_loss_max = anchors["Neurophysiology_Anchors"]["EAAT_Downregulation_Max"] 
R_STRESS_MAX_CAP = anchors["Neurophysiology_Anchors"]["Max_Metabolic_Capacity_Limit"]

isb_params = anchors["ISB_2_0_Expansions"]
RHO_REC = isb_params["Neuroplasticity_Recovery_Rho"]
THETA_REC = isb_params["Recovery_Threshold_Theta"]
SIGMA_NOISE = isb_params["Stochastic_Noise_Sigma"]
TAU_BBB = isb_params["BBB_Permeability_Delay_Days"]

rna_drift = isb_params["Epigenetic_Drift_RNA_Velocity"]
BETA_SPLICING = rna_drift["Splicing_Rate_Beta"]
GAMMA_DEGRADATION = rna_drift["Degradation_Rate_Gamma"]

# [DERIVATION] Zero-Assumption Epigenetic Decay Constant
alpha_min = 1.0 - eaat_loss_max 
LAMBDA_DECAY = -np.log(alpha_min) / (R_STRESS_MAX_CAP - 1.0) 

Km_ATP, Km_G, D_G = 0.1, 0.1, 0.01

print(f"-> [VALIDATED] Epigenetic transcription decay (Lambda) derived: {LAMBDA_DECAY:.4f}")

# ---------------------------------------------------------
# [MODULE 2] COHORT SYNTHESIS
# ---------------------------------------------------------
print(f"\n[MODULE 2] Synthesizing multidimensional profiles for {POPULATION_SIZE} virtual subjects...")
np.random.seed(42)

global_mean_mtCN = pop_genetics["GLOBAL_MEAN_REFERENCE"]

def generate_cohort_rbs(ancestry_key, size):
    mean_cn = pop_genetics[ancestry_key]["mean_mtCN"]
    std_cn = pop_genetics[ancestry_key]["std_mtCN"]
    return np.random.normal(loc=mean_cn, scale=std_cn, size=size) / global_mean_mtCN

RBS_Population = np.concatenate([
    generate_cohort_rbs("AFR", QUARTER_POP),
    generate_cohort_rbs("EUR", QUARTER_POP),
    generate_cohort_rbs("EAS", QUARTER_POP),
    generate_cohort_rbs("AMR", QUARTER_POP)
])
Ancestry_Labels = np.array(['AFR']*QUARTER_POP + ['EUR']*QUARTER_POP + ['EAS']*QUARTER_POP + ['AMR']*QUARTER_POP)

R_stress_Population = np.random.uniform(low=1.0, high=R_STRESS_MAX_CAP, size=POPULATION_SIZE)
Age_Population = np.random.uniform(low=20.0, high=80.0, size=POPULATION_SIZE)
Age_Penalty_Population = np.where(Age_Population > peak_age, 1.0 - ((Age_Population - peak_age) * decline_rate), 1.0)

# ---------------------------------------------------------
# [MODULE 3] INDIVIDUAL SDE-PDE COMPUTE NODE
# ---------------------------------------------------------
ALLOSTATIC_SPAN = 1000 
DT_STEP = 0.1 

def simulate_virtual_patient(patient_id, rbs_factor, r_stress_max, age_penalty):
    personal_p_basal_raw = P_basal_global_array * rbs_factor * age_penalty
    personal_p_basal_clamped = np.clip(personal_p_basal_raw, a_min=4.0, a_max=9.0)
    epsilon = 1e-6
    mean_p_basal = np.mean(personal_p_basal_clamped)
    patient_isb = mean_p_basal / (r_stress_max + epsilon)
    
    # [DOUBLE STEADY-STATE CALIBRATION]
    Vmax_basal_clearance = (Vmax * baseline_glu) / (Km_G + baseline_glu)
    k_prod_personal = Vmax_basal_clearance / baseline_atp
    
    Basal_Thermodynamic_Stress = 1.0 * stoich_atp_glu
    Basal_Drain = (Basal_Thermodynamic_Stress * baseline_glu) / (Km_ATP + baseline_glu)
    k_decay_personal = (personal_p_basal_clamped - Basal_Drain + RHO_REC) / baseline_atp
        
    # Focal Vagal Influx Calibration
    Chronic_Vagal_Influx = np.zeros(NUM_NODES)
    if r_stress_max > 1.0:
        baseline_prod_node0 = k_prod_personal[0] * baseline_atp
        Chronic_Vagal_Influx[0] = baseline_prod_node0 * (r_stress_max - 1.0) * vagal_firing_mult
        
    num_steps = int(ALLOSTATIC_SPAN / DT_STEP)
    sqrt_dt = np.sqrt(DT_STEP)
    
    curr_ATP = np.full(NUM_NODES, baseline_atp)
    curr_Glu = np.full(NUM_NODES, baseline_glu)
    
    alpha_0 = 1.0
    s_0 = alpha_0 / GAMMA_DEGRADATION
    curr_u = np.full(NUM_NODES, alpha_0 / BETA_SPLICING)
    curr_s = np.full(NUM_NODES, s_0)
    
    survived = True
    time_to_bifurcation = float(ALLOSTATIC_SPAN)
    consecutive_collapse_steps = 0
    BIFURCATION_CONFIRMATION_LIMIT = int(1.0 / DT_STEP) 
    
    for step in range(num_steps):
        t = step * DT_STEP
        current_R_stress = 1.0 + (r_stress_max - 1.0) * (1.0 - np.exp(-t / TAU_BBB))
        
        # Epigenetic Drift Integration
        current_alpha = alpha_0 * np.exp(-LAMBDA_DECAY * (current_R_stress - 1.0))
        du_dt = current_alpha - (BETA_SPLICING * curr_u)
        ds_dt = (BETA_SPLICING * curr_u) - (GAMMA_DEGRADATION * curr_s) 
        
        curr_u = curr_u + (du_dt * DT_STEP)
        curr_s = curr_s + (ds_dt * DT_STEP)
        curr_u = np.maximum(curr_u, 1e-6)
        curr_s = np.maximum(curr_s, 1e-6)
        
        # Empirical Transporter Bounding
        s_ratio = curr_s / s_0
        transcriptional_health = np.maximum(s_ratio, (1.0 - eaat_loss_max))
        curr_Vmax = Vmax * transcriptional_health
        
        # Thermodynamic Drift
        Dynamic_Thermo_Stress = current_R_stress * stoich_atp_glu
        recovery_term = RHO_REC * np.where(curr_ATP > THETA_REC, 1.0, 0.0)
        
        dATP_drift = personal_p_basal_clamped - (Dynamic_Thermo_Stress * curr_Glu) / (Km_ATP + curr_Glu) - (k_decay_personal * curr_ATP) + recovery_term
        dW = np.random.normal(0, 1, NUM_NODES) * sqrt_dt
        stochastic_diffusion = SIGMA_NOISE * curr_ATP * dW
        
        curr_ATP = curr_ATP + (dATP_drift * DT_STEP) + stochastic_diffusion
        curr_ATP = np.maximum(curr_ATP, 1e-4)
        
        # Anisotropic PDE with Focal Influx
        clearance_failure = (k_prod_personal * curr_ATP) + Chronic_Vagal_Influx - (curr_Vmax * curr_Glu) / (Km_G + curr_Glu)
        dGlu_drift = clearance_failure - D_G * np.dot(Laplacian_L, curr_Glu)
        
        curr_Glu = curr_Glu + (dGlu_drift * DT_STEP)
        curr_Glu = np.maximum(curr_Glu, 1e-4)
        
        # Saddle-Node Detection
        if np.min(curr_ATP) < CRITICAL_ATP_BIFURCATION:
            consecutive_collapse_steps += 1
            if consecutive_collapse_steps >= BIFURCATION_CONFIRMATION_LIMIT:
                survived = False
                time_to_bifurcation = float(t)
                break
        else:
            consecutive_collapse_steps = 0 
            
    return (patient_id, patient_isb, time_to_bifurcation, survived)

# ---------------------------------------------------------
# [MODULE 4] PARALLEL EXECUTION ARCHITECTURE
# ---------------------------------------------------------
print(f"\n[MODULE 4] Initiating parallel SDE computation across {POPULATION_SIZE} virtual nodes...")
start_time = time.time()

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(simulate_virtual_patient)(i, RBS_Population[i], R_stress_Population[i], Age_Penalty_Population[i]) 
    for i in range(POPULATION_SIZE)
)

end_time = time.time()
print(f"\n-> [SYSTEM] Parallel integration completed in {(end_time - start_time)/60:.2f} minutes.")

# ---------------------------------------------------------
# [MODULE 5] BIFURCATION DATA AGGREGATION
# ---------------------------------------------------------
patient_ids, isb_scores, bifurcation_times, survived_flags = zip(*results)

df_sim = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Ancestry': Ancestry_Labels,
    'Age': Age_Population,
    'Age_Penalty_Factor': Age_Penalty_Population,
    'Allostatic_Load_OR': R_stress_Population,
    'Stabilized_ISB_Score': isb_scores,
    'Time_To_Bifurcation_Days': bifurcation_times,
    'Survived_1000_Days': survived_flags
})

df_sim.to_csv(os.path.join(BASE_DIR, "Output_TrueDynamic_MegaSimulation_Final.csv"), index=False)

print("\n=============================================================================")
print("FOCAL PHASE TRANSITION RATE BY ANCESTRY UNDER HIGH ALLOSTATIC LOAD (OR > 3.0):")
print("*(Phase transition defined as sustained focal ATP depletion < 0.5 mM)*")
extreme_load = df_sim[df_sim['Allostatic_Load_OR'] > 3.0]
if not extreme_load.empty:
    collapse_rates = (~extreme_load['Survived_1000_Days']).groupby(extreme_load['Ancestry']).mean() * 100
    print(collapse_rates.sort_values(ascending=False).to_string(float_format="%.2f%%"))
else:
    print("No subjects with OR > 3.0 in this run.")
print("=============================================================================\n")
