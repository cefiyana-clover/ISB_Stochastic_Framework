"""
=============================================================================
PIPELINE ISB (FIRST-PRINCIPLES Q1 STANDARD) - SCRIPT 03: MULTI-GPU (T4 x2)
=============================================================================
Hardware Acceleration & Mathematical Isomorphism:
1. Executes the exact Patankar Semi-Implicit First-Principles SDE-PDE engine
   in 64-bit Double Precision (torch.float64) on NVIDIA CUDA GPUs (T4 x2).
2. Automatically distributes subject batches across BOTH GPUs (cuda:0 and
   cuda:1) concurrently via ThreadPoolExecutor (since PyTorch CUDA kernels
   release the Python GIL), completing N = 100,000 subjects per regime in
   ~1 to 2 minutes instead of ~6 hours on CPU.
3. Zero host-device synchronization inside the 10,000-step time loop.
=============================================================================
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

print("[SYSTEM] Initializing Multi-GPU (CUDA T4 x2) First-Principles Monte Carlo Engine...")

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.path.abspath('.')

BASE_DIR = os.path.join(CURRENT_DIR, 'ISB_Empirical_Vault')
os.makedirs(BASE_DIR, exist_ok=True)

# Detect available CUDA GPUs (T4 x2 on Kaggle provides 2 GPUs)
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 0:
    DEVICES = [f"cuda:{i}" for i in range(NUM_GPUS)]
    for i in range(NUM_GPUS):
        print(f"-> [GPU DETECTED] Device cuda:{i} — {torch.cuda.get_device_name(i)}")
else:
    DEVICES = ["cpu"]
    print("-> [WARNING] No CUDA GPU detected. Falling back to CPU.")

# [COMPUTE LOCK] Full 100,000 Virtual Subjects per Regime
POPULATION_SIZE = 100000
QUARTER_POP = POPULATION_SIZE // 4
GPU_BATCH_SIZE = 5000     # 5,000 subjects per GPU kernel batch (~6 MB VRAM per tensor in float64)
ALLOSTATIC_SPAN = 1000
DT_STEP = 0.1             # Strictly synchronized with Script 02 and Manuscript Section 2.2 (0.1 day)

# ---------------------------------------------------------
# [MODULE 1] EMPIRICAL MATRIX & BIOPHYSICAL ANCHOR INGESTION
# ---------------------------------------------------------
L2_DTI_W = np.load(os.path.join(BASE_DIR, "L2_DTI_Structural_Mask.npy"))
P_basal_global = np.load(os.path.join(BASE_DIR, "L3_Transcriptomic_Pbasal.npy"))
Vmax_baseline = np.load(os.path.join(BASE_DIR, "L4_PET_Vmax_Density.npy"))

with open(os.path.join(BASE_DIR, "L1_Destrieux_ROI_Metadata.json"), "r") as f:
    roi_meta = json.load(f)
with open(os.path.join(BASE_DIR, "L5_L6_Clinical_Anchors.json"), "r") as f:
    anchors = json.load(f)
with open(os.path.join(BASE_DIR, "L7_Population_Genetics.json"), "r") as f:
    pop_tiers = json.load(f)

NUM_NODES = int(L2_DTI_W.shape[0])
Degree_Vector = np.sum(L2_DTI_W, axis=1)[:, None]

mrs = anchors["Multimodal_31P_1H_MRS_Anchors"]
kin = anchors["Bistable_Kinetic_Constants"]
isb = anchors["ISB_2_0_Expansions"]
epi = isb["Epigenetic_Chromatin_SLC1A2"]
sen = anchors["Mitochondrial_Senescence"]
neuro = anchors["Neurophysiology_Anchors"]

# Verified Empirical Biophysical Constants (Audit Table 3.3)
A0 = float(mrs["Baseline_ATP_mM"])                        # 3.0 mM (Magistretti & Allaman, 2015)
G0 = float(mrs["Baseline_Glutamate_mM"])                  # 0.002 mM (Herman & Jahr, 2007)
THETA_CRIT = float(mrs["Saddle_Node_Threshold"])          # 0.50 mM (Erecinska & Silver, 1994)

Km_G = float(kin["Km_G_EAAT2_mM"])                        # 0.020 mM (Arriza et al., 1994; Danbolt, 2001)
Km_ATP = float(kin["Km_ATP_NaKATPase_mM"])                # 0.400 mM (Blanco & Mercer, 1998)
Km_auto = float(kin["Km_Auto_Glycolysis_mM"])             # 0.450 mM (Chandra et al., 2011; Jolivet et al., 2015)
stoich = float(kin["ATP_per_Glutamate_Stoichiometry"])    # 2.0 mol ATP/mol Glu (Pellerin & Magistretti, 1994)
k_stress = float(kin["Allostatic_Stress_Coupling_k"])     # 0.62 mM/day
k_spill = float(kin["Extracellular_Spillover_Clearance_k"]) # 0.15 day^-1
D_G = float(kin["Network_Diffusion_DG"])                  # 0.015 day^-1

RHO_GLYC = float(isb["Glycogen_Rescue_Rho"])              # 0.35 mM/day (Brown, 2004; Dienel, 2019)
THETA_GLYC = float(isb["Crisis_Threshold_Theta_mM"])      # 1.20 mM (Brown, 2004)
SIGMA_A = float(isb["Stochastic_Noise_Sigma_ATP"])        # 0.065 day^-1/2
SIGMA_G = float(isb["Stochastic_Noise_Sigma_Glu"])        # 0.025 day^-1/2
TAU_BBB = float(isb["BBB_Allostatic_Delay_Days"])         # 45.0 days (Menard et al., 2017)

k_meth = float(epi["Methylation_Repression_Rate_k_meth"])     # 0.0045 day^-1 (Bintu et al., 2016)
k_demeth = float(epi["Demethylation_Recovery_Rate_k_demeth"]) # 0.0060 day^-1 (Bintu et al., 2016)
BETA_EFF = float(epi["Effective_Maturation_Rate_Beta"])       # 0.25 day^-1 (Schwanhausser et al., 2011)
GAMMA_EFF = float(epi["Effective_Turnover_Rate_Gamma"])       # 0.12 day^-1 (Danbolt, 2001)

eaat_loss_max = float(neuro["EAAT_Downregulation_Max"])       # 0.50 (Zschocke et al., 2005)
vagal_mult = float(neuro["Vagal_Hyperfiring_Multiplier"])     # 1.85

C_GLYC_POOL = stoich * A0
K_SYN_GLYC = GAMMA_EFF

primary_insula_nodes = [int(i) for i in roi_meta["primary_insula_nodes"]]
secondary_insula_nodes = [int(i) for i in roi_meta["secondary_insula_nodes"]]

# ---------------------------------------------------------
# [MODULE 2] EXACT FIRST-PRINCIPLES STEADY-STATE CALIBRATION
# ---------------------------------------------------------
V_influx_weights = np.zeros(NUM_NODES, dtype=np.float64)
for idx in primary_insula_nodes:
    V_influx_weights[idx] = 0.35
for idx in secondary_insula_nodes:
    V_influx_weights[idx] = 0.15

V_influx_col = V_influx_weights[:, None]
Vmax_col = Vmax_baseline[:, None]
P_basal_col = P_basal_global[:, None]

f_auto_0 = (A0 ** 2) / (Km_auto ** 2 + A0 ** 2)
f_atp_0 = (A0 ** 2) / (Km_ATP ** 2 + A0 ** 2)
f_glu_0 = G0 / (Km_G + G0)
f_glyc_0 = RHO_GLYC * ((THETA_GLYC ** 4) / (THETA_GLYC ** 4 + A0 ** 4)) * f_auto_0

J_EAAT_0 = Vmax_col * f_atp_0 * f_glu_0
J_syn_0 = J_EAAT_0 + k_spill * G0

P_glyc_basal = P_basal_col * f_glu_0
P_max_ref = (P_basal_col * (1.0 - f_glu_0)) / f_auto_0
Basal_ATP_Drain = stoich * (J_EAAT_0 + k_stress * 1.0 * f_glu_0)
k_house_ref = (P_basal_col - Basal_ATP_Drain + f_glyc_0) / A0

mean_cortical_pbasal = float(np.mean(P_basal_global))

# ---------------------------------------------------------
# [MODULE 3] STATISTICALLY INDEPENDENT MONTE CARLO COHORT SYNTHESIS
# ---------------------------------------------------------
print(f"\n[MODULE 3] Synthesizing independent multidimensional profiles for {POPULATION_SIZE} virtual subjects...")
rng_master = np.random.default_rng(42)
global_ref_cn = float(pop_tiers["GLOBAL_MEAN_REFERENCE"])
peak_age = float(sen["Peak_Optimal_Age"])
decline_rate = float(sen["Decline_Rate_Per_Year"])

tier_keys = ["Tier_IV_Low", "Tier_I_High", "Tier_III_ModLow", "Tier_II_ModHigh"]
rbs_list, tier_labels, legacy_labels = [], [], []

for t_key in tier_keys:
    info = pop_tiers[t_key]
    samples = rng_master.normal(loc=info["mean_mtCN"], scale=info["std_mtCN"], size=QUARTER_POP)
    rbs_list.append(np.clip(samples / global_ref_cn, 0.65, 1.35))
    tier_labels.extend([t_key] * QUARTER_POP)
    legacy_labels.extend([info["legacy_code"]] * QUARTER_POP)

RBS_Population = np.concatenate(rbs_list)
Tier_Labels = np.array(tier_labels)
Ancestry_Labels = np.array(legacy_labels)

R_stress_Population = rng_master.uniform(1.0, 4.0, size=POPULATION_SIZE)
Age_Population = rng_master.uniform(20.0, 80.0, size=POPULATION_SIZE)
Age_Penalty_Population = np.where(
    Age_Population > peak_age,
    1.0 - (Age_Population - peak_age) * decline_rate,
    1.0
)

# ---------------------------------------------------------
# [MODULE 4] PYTORCH CUDA (FLOAT64) PATANKAR BATCH SOLVER
# ---------------------------------------------------------
@torch.no_grad()
def simulate_gpu_batch_first_principles(idx_start, idx_end, regime, device_str):
    device = torch.device(device_str)
    batch_indices = np.arange(idx_start, idx_end)
    B_sub = len(batch_indices)

    # Move static anatomical & kinetic vectors to target GPU in 64-bit Double Precision
    W_t = torch.tensor(L2_DTI_W, dtype=torch.float64, device=device)
    Deg_t = torch.tensor(Degree_Vector, dtype=torch.float64, device=device)
    V_inf_t = torch.tensor(V_influx_col, dtype=torch.float64, device=device)
    Vmax_t = torch.tensor(Vmax_col, dtype=torch.float64, device=device)
    J_syn_0_t = torch.tensor(J_syn_0, dtype=torch.float64, device=device)
    P_glyc_t = torch.tensor(P_glyc_basal, dtype=torch.float64, device=device)
    P_max_ref_t = torch.tensor(P_max_ref, dtype=torch.float64, device=device)
    k_house_t = torch.tensor(k_house_ref, dtype=torch.float64, device=device)

    # Subject-specific parameters on GPU
    rbs_b = torch.tensor(RBS_Population[batch_indices][None, :], dtype=torch.float64, device=device)
    stress_b = torch.tensor(R_stress_Population[batch_indices][None, :], dtype=torch.float64, device=device)
    age_pen_b = torch.tensor(Age_Penalty_Population[batch_indices][None, :], dtype=torch.float64, device=device)

    bio_reserve_b = rbs_b * age_pen_b
    isb_scores_b = (mean_cortical_pbasal * bio_reserve_b[0, :]) / (stress_b[0, :] + 1e-9)
    P_max_patient_b = P_max_ref_t * bio_reserve_b

    num_steps = int(ALLOSTATIC_SPAN / DT_STEP)
    sqrt_dt = float(np.sqrt(DT_STEP))
    confirm_limit = int(2.0 / DT_STEP)

    curr_A = torch.full((NUM_NODES, B_sub), A0, dtype=torch.float64, device=device)
    curr_G = torch.full((NUM_NODES, B_sub), G0, dtype=torch.float64, device=device)
    curr_E = torch.ones((NUM_NODES, B_sub), dtype=torch.float64, device=device)
    curr_R = torch.ones((NUM_NODES, B_sub), dtype=torch.float64, device=device)

    u0 = 1.0 / BETA_EFF
    s0 = 1.0 / GAMMA_EFF
    curr_u = torch.full((NUM_NODES, B_sub), u0, dtype=torch.float64, device=device)
    curr_s = torch.full((NUM_NODES, B_sub), s0, dtype=torch.float64, device=device)

    omega_eff_b = torch.ones((1, B_sub), dtype=torch.float64, device=device)
    survived_b = torch.ones(B_sub, dtype=torch.bool, device=device)
    t_bif_b = torch.full((B_sub,), float(ALLOSTATIC_SPAN), dtype=torch.float64, device=device)
    collapse_cnt_b = torch.zeros(B_sub, dtype=torch.int32, device=device)
    zero_int = torch.zeros(B_sub, dtype=torch.int32, device=device)

    # Isolated CUDA Generator per batch for reproducible independent Wiener paths
    seed_base = 100000 if regime == "REGIME_I" else 500000
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_base + int(idx_start))

    Km_ATP_sq = Km_ATP ** 2
    Km_auto_sq = Km_auto ** 2
    THETA_GLYC_4 = THETA_GLYC ** 4
    min_E_bound = 1.0 - eaat_loss_max

    for step in range(num_steps):
        t = step * DT_STEP
        # 1. Blood-Brain Barrier Exponential Transit
        omega_eff_b.add_(((stress_b - omega_eff_b) / TAU_BBB) * DT_STEP)
        stress_excess_b = torch.clamp(omega_eff_b - 1.0, min=0.0)

        # 2. Chromatin Remodeling E_i(t) & Transcript Turnover
        dE_dt = (k_demeth * bio_reserve_b) * (1.0 - curr_E) - k_meth * stress_excess_b * (curr_E - min_E_bound)
        curr_E = torch.clamp(curr_E + dE_dt * DT_STEP, min=min_E_bound, max=1.0)

        du_dt = curr_E - BETA_EFF * curr_u
        ds_dt = BETA_EFF * curr_u - GAMMA_EFF * curr_s
        curr_u = torch.clamp(curr_u + du_dt * DT_STEP, min=1e-6)
        curr_s = torch.clamp(curr_s + ds_dt * DT_STEP, min=1e-6)

        Vmax_dyn = Vmax_t * torch.clamp(curr_s / s0, min=min_E_bound, max=1.0)

        # 3. Cooperative Hill ATP-Dependent EAAT2 Flux & Autocatalytic Synthesis
        curr_A_sq = curr_A * curr_A
        f_atp = curr_A_sq / (Km_ATP_sq + curr_A_sq)
        f_auto = curr_A_sq / (Km_auto_sq + curr_A_sq)
        f_glu = curr_G / (Km_G + curr_G)

        J_EAAT = Vmax_dyn * f_atp * f_glu

        # 4. Emergency Glycogenolysis J_glyc & Mass-Balance Conservation
        crisis_hill = THETA_GLYC_4 / (THETA_GLYC_4 + curr_A_sq * curr_A_sq)
        if regime == "REGIME_I":
            J_glyc = RHO_GLYC * curr_R * crisis_hill * f_auto
            glycogen_shield = 1.0
        else:
            J_glyc = RHO_GLYC * curr_R * crisis_hill
            glycogen_mobilization = J_glyc * (1.0 + stress_excess_b * V_inf_t)
            dR_dt = (K_SYN_GLYC * bio_reserve_b) * f_auto * (1.0 - curr_R) - (glycogen_mobilization / C_GLYC_POOL)
            curr_R = torch.clamp(curr_R + dR_dt * DT_STEP, min=0.0, max=1.0)
            glycogen_shield = curr_R

        # 5. Patankar Semi-Implicit Bistable ATP SDE
        local_allostatic_drive = 1.0 + V_inf_t * stress_excess_b * vagal_mult * (stoich / (glycogen_shield + Km_ATP))
        J_stress_ionic = k_stress * local_allostatic_drive * f_glu

        atp_source = P_glyc_t + P_max_patient_b * f_auto + J_glyc
        active_drain_flux = stoich * (J_EAAT + J_stress_ionic * f_atp)
        atp_sink_rate = k_house_t + (active_drain_flux / (curr_A + Km_G))

        dW_A = torch.randn((NUM_NODES, B_sub), dtype=torch.float64, device=device, generator=gen) * sqrt_dt
        numerator_A = curr_A + atp_source * DT_STEP + SIGMA_A * curr_A * dW_A
        denominator_A = 1.0 + atp_sink_rate * DT_STEP
        curr_A = torch.clamp(numerator_A / denominator_A, min=1e-4)

        # 6. Patankar Semi-Implicit Extracellular Glutamate Reaction-Diffusion PDE
        vagal_t = J_syn_0_t * V_inf_t * stress_excess_b * vagal_mult
        pump_deficit_spillover = J_syn_0_t * V_inf_t * (1.0 - f_atp) * stress_excess_b
        diffusive_inflow = D_G * torch.matmul(W_t, curr_G)

        dW_G = torch.randn((NUM_NODES, B_sub), dtype=torch.float64, device=device, generator=gen) * sqrt_dt
        numerator_G = (
            curr_G
            + (J_syn_0_t + vagal_t + pump_deficit_spillover + diffusive_inflow) * DT_STEP
            + SIGMA_G * curr_G * dW_G
        )
        denominator_G = 1.0 + (
            (Vmax_dyn * f_atp) / (Km_G + curr_G)
            + k_spill
            + D_G * Deg_t
        ) * DT_STEP
        curr_G = torch.clamp(numerator_G / denominator_G, min=1e-5)

        # 7. Zero-Sync GPU Bifurcation Detection (< 0.50 mM for >= 20 consecutive steps)
        min_A_per_subj = torch.min(curr_A, dim=0).values
        breached = (min_A_per_subj < THETA_CRIT) & survived_b
        collapse_cnt_b = torch.where(breached, collapse_cnt_b + 1, zero_int)

        newly_collapsed = survived_b & (collapse_cnt_b >= confirm_limit)
        t_bif_b = torch.where(newly_collapsed, float(t), t_bif_b)
        survived_b = survived_b & (~newly_collapsed)

    min_r_per_subj = torch.min(curr_R, dim=0).values
    return (
        batch_indices,
        isb_scores_b.cpu().numpy(),
        t_bif_b.cpu().numpy(),
        survived_b.cpu().numpy(),
        min_r_per_subj.cpu().numpy()
    )

# ---------------------------------------------------------
# [MODULE 5] DUAL-GPU (T4 x2) EXECUTION & CSV ARCHIVING
# ---------------------------------------------------------
if __name__ == "__main__":
    batch_ranges = [
        (i, min(POPULATION_SIZE, i + GPU_BATCH_SIZE))
        for i in range(0, POPULATION_SIZE, GPU_BATCH_SIZE)
    ]

    for regime_name, csv_filename in [
        ("REGIME_I", "Output_TrueDynamic_MegaSimulation_Final.csv"),
        ("REGIME_II", "Output_Regime2_MegaSimulation_Final.csv")
    ]:
        print(f"\n=============================================================================")
        print(f"EXECUTING MULTI-GPU MONTE CARLO ({POPULATION_SIZE} SUBJECTS): {regime_name}")
        print(f"Active GPUs: {len(DEVICES)} ({DEVICES}) | Total Batches: {len(batch_ranges)} (Batch Size = {GPU_BATCH_SIZE})")
        print(f"=============================================================================")
        start_time = time.time()

        # Dispatch batches concurrently across cuda:0 and cuda:1
        tasks = [
            (b_start, b_end, regime_name, DEVICES[idx % len(DEVICES)])
            for idx, (b_start, b_end) in enumerate(batch_ranges)
        ]

        batch_outputs = []
        with ThreadPoolExecutor(max_workers=len(DEVICES)) as executor:
            futures = [executor.submit(simulate_gpu_batch_first_principles, *t) for t in tasks]
            for idx_f, fut in enumerate(futures):
                res = fut.result()
                batch_outputs.append(res)
                print(f"   -> [{regime_name}] Completed GPU Batch {idx_f + 1}/{len(tasks)} "
                      f"({(idx_f + 1) * GPU_BATCH_SIZE}/{POPULATION_SIZE} subjects | {(time.time() - start_time):.1f}s elapsed)")

        all_ids = np.concatenate([out[0] for out in batch_outputs])
        order = np.argsort(all_ids)
        pids = all_ids[order]
        isbs = np.concatenate([out[1] for out in batch_outputs])[order]
        t_bifs = np.concatenate([out[2] for out in batch_outputs])[order]
        survs = np.concatenate([out[3] for out in batch_outputs])[order]
        min_rs = np.concatenate([out[4] for out in batch_outputs])[order]

        elapsed_sec = time.time() - start_time
        print(f"\n-> [SYSTEM] {regime_name} ({POPULATION_SIZE} subjects) completed in {elapsed_sec:.2f} seconds ({elapsed_sec/60.0:.2f} min).")

        df_out = pd.DataFrame({
            "Patient_ID": pids,
            "Polygenic_Tier": Tier_Labels,
            "Ancestry": Ancestry_Labels,
            "Age": Age_Population,
            "Age_Penalty_Factor": Age_Penalty_Population,
            "Allostatic_Load_OR": R_stress_Population,
            "Stabilized_ISB_Score": isbs,
            "Min_Recovery_Capacitor": min_rs,
            "Time_To_Bifurcation_Days": t_bifs,
            "Survived_1000_Days": survs
        })

        csv_path = os.path.join(BASE_DIR, csv_filename)
        df_out.to_csv(csv_path, index=False)

        total_collapse_pct = (~df_out["Survived_1000_Days"]).mean() * 100.0
        collapsed_df = df_out[~df_out["Survived_1000_Days"]]
        mean_collapse_age = collapsed_df["Age"].mean() if not collapsed_df.empty else 0.0
        young_overload = df_out[(df_out["Age"] <= 35.0) & (df_out["Allostatic_Load_OR"] > 3.2)]
        young_collapse_pct = (~young_overload["Survived_1000_Days"]).mean() * 100.0 if not young_overload.empty else 0.0
        r_load_isb = np.corrcoef(df_out["Allostatic_Load_OR"], df_out["Stabilized_ISB_Score"])[0, 1]
        r_age_isb = np.corrcoef(df_out["Age"], df_out["Stabilized_ISB_Score"])[0, 1]

        print("\n-----------------------------------------------------------------------------")
        print(f"[{regime_name}] TOTAL POPULATION BIFURCATION RATE (ALL LOADS): {total_collapse_pct:.2f}%")
        print(f"[{regime_name}] MEAN AGE AT BIFURCATION: {mean_collapse_age:.2f} years | r(Load, ISB) = {r_load_isb:.2f} | r(Age, ISB) = {r_age_isb:.2f}")
        print(f"[{regime_name}] YOUNG ADULT (AGE 20-35) BIFURCATION RATE UNDER EXTREME OR (>3.2): {young_collapse_pct:.2f}%")
        print(f"[{regime_name}] FOCAL PHASE TRANSITION RATE BY POLYGENIC TIER UNDER OR > 3.0:")
        extreme_load = df_out[df_out["Allostatic_Load_OR"] > 3.0]
        if not extreme_load.empty:
            summary_table =Extreme_summary = extreme_load.groupby(["Polygenic_Tier", "Ancestry"]).agg(
                Sub_Population=("Patient_ID", "count"),
                Collapse_Rate_Pct=("Survived_1000_Days", lambda s: (~s).mean() * 100.0),
                Median_Time_To_Bifurcation_Among_Collapsed=(
                    "Time_To_Bifurcation_Days", lambda s: s[s < 999.9].median()
                )
            ).sort_values(by="Collapse_Rate_Pct", ascending=False)
            print(summary_table.to_string(float_format="%.2f"))
        print("=============================================================================\n")
