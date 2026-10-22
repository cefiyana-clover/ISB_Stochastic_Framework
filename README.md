# The Bioenergetic Stability Index (ISB): A Multi-Scale Stochastic-Spatiotemporal Framework of Focal Phase Transitions in Major Depressive Phenotypes

![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)
![Precision](https://img.shields.io/badge/Compute-PyTorch_CUDA_Float64-blue)
![Atlas](https://img.shields.io/badge/Connectome-Destrieux_148_ROIs-orange)
![License: GPL v3](https://img.shields.io/badge/Code_License-GPLv3-blue.svg)
![License: CC BY-NC 4.0](https://img.shields.io/badge/Theory_License-CC_BY--NC_4.0-lightgrey.svg)

This repository houses the official high-performance computational architecture for **The Bioenergetic Stability Index (ISB)**. The framework models major depressive phenotypes as a macroscopic manifestation of focal astrocytic bioenergetic phase transitions driven by cumulative allostatic load, slow glucocorticoid-mediated epigenetic repression of *SLC1A2* (EAAT2), finite glycogenolytic mass-balance capacitance, and intrinsic biological noise across a 148-node callosal-constrained cortical connectome.

---

## 1. First-Principles Biophysical Architecture

Unlike monostable linear rate models ($\operatorname{Tr}(J) < 0, \det(J) > 0$), the ISB engine formulates a dimensionally homogeneous, genuinely bistable coupled system of Itô Stochastic and Partial Differential Equations (SDE-PDE) integrated via an unconditionally positivity-preserving **Patankar-type semi-implicit Euler-Maruyama scheme** ($\Delta t = 0.1\text{ day}$).

### A. Cooperative Hill Kinetics & Coupled Bistable SDE-PDE System
Active clearance of extracellular glutamate ($G_i$, initialized at $G_0 = 0.002\text{ mM}$) by astrocytic Excitatory Amino Acid Transporter 2 (EAAT2, encoded by *SLC1A2*) relies strictly on the transmembrane $\text{Na}^+$ gradient maintained by the astrocytic $\alpha_2\beta_2\text{ Na}^+/\text{K}^+$-ATPase (*ATP1A2*), governed by second-order Hill cooperativity on cytosolic ATP ($A_i$, initialized at $A_0 = 3.0\text{ mM}$):

$$f_{\text{atp}}(A_i) = \frac{A_i^2}{K_{m,\text{ATP}}^2 + A_i^2}, \quad f_{\text{auto}}(A_i) = \frac{A_i^2}{K_{m,\text{auto}}^2 + A_i^2}, \quad f_{\text{glu}}(G_i) = \frac{G_i}{K_{m,G} + G_i}, \quad \chi_{\text{crisis}}(A_i) = \frac{\theta_{\text{glyc}}^4}{\theta_{\text{glyc}}^4 + A_i^4}$$

$$J_{\text{EAAT},i}(A_i, G_i, t) = V_{\max,i}^*(t) \, f_{\text{atp}}(A_i) \, f_{\text{glu}}(G_i)$$

where $K_{m,\text{ATP}} = 0.400\text{ mM}$, $K_{m,\text{auto}} = 0.450\text{ mM}$, $K_{m,G} = 0.020\text{ mM}$ ($20.0\ \mu\text{M}$), and $\theta_{\text{glyc}} = 1.20\text{ mM}$. The coupled Itô SDE-PDE equations governing cytosolic astrocytic ATP ($A_i(t)$) and extracellular synaptic/perisynaptic glutamate ($G_i(t)$) across each cortical parcel $i \in \{1, \dots, 148\}$ are:

$$dA_i = \left[ P_{\text{glyc},i}^0 + P_{\max,i}^{(m)} f_{\text{auto}}(A_i) + J_{\text{glyc},i}(A_i, R_{\text{rec},i}) - \sigma_{\text{stoich}}\left(J_{\text{EAAT},i}(A_i, G_i, t) + J_{\text{stress},i}(G_i, t) f_{\text{atp}}(A_i)\right) - k_{\text{house},i} A_i \right] dt + \sigma_A A_i \, dW_{A,i}(t)$$

$$dG_i = \left[ J_{\text{syn},i}^0 + V_{\text{influx},i}(t) + J_{\text{spillover},i}(A_i, t) - J_{\text{EAAT},i}(A_i, G_i, t) - k_{\text{spill}} G_i - D_G \sum_{j=1}^{148} L_{ij} G_j \right] dt + \sigma_G G_i \, dW_{G,i}(t)$$

where $\sigma_{\text{stoich}} = 2.0\text{ mol ATP/mol Glu}$ represents the canonical Astrocyte-Neuron Lactate Shuttle (ANLS) and Glutamate-Glutamine cycle stoichiometry ($1.0\text{ ATP}$ for $\text{Na}^+/\text{K}^+$-ATPase extrusion plus $1.0\text{ ATP}$ for Glutamine Synthetase amidation). As chronic allostatic load erodes the local Jacobian determinant toward the fold bifurcation boundary ($\det(J) = 0, \operatorname{Tr}(J) < 0$), multiplicative Langevin fluctuations ($\sigma_A = 0.065\text{ day}^{-1/2}, \sigma_G = 0.025\text{ day}^{-1/2}$) drive Kramers stochastic escape across the critical electrogenic pump failure threshold ($\theta_c = 0.50\text{ mM}$).

### B. Callosal-Constrained Connectome & Bilateral Insular Anchoring
Cortical topology is parcellated into $N = 148$ bilateral gyral and sulcal regions defined by the Destrieux cortical atlas (`aparc.a2009s`).
* **Viscerosensory Insular Epicenters:** Ascending vagal-NTS-parabrachial-thalamo-insular projections are anchored bilaterally at the primary posterior interoceptive insula (`L & R G_Ins_lg_and_S_cent_ins`, node indices `16` and `90`, $w_i = 0.35$ per hemisphere) and secondary anterior visceromotor insula (`L & R G_insular_short`, node indices `17` and `91`, $w_i = 0.15$ per hemisphere).
* **Corpus Callosum Waypoint Routing:** Inter-hemispheric axonal distances $d_{\text{eff}}(i, j)$ are routed through MNI152 corpus callosum waypoints (Genu $[0, +26, +6]\text{ mm}$, Body $[0, -10, +24]\text{ mm}$, and Splenium $[0, -34, +14]\text{ mm}$) weighted by a homotopic symmetry mask to eliminate artificial cross-fissure Euclidean short-circuiting prior to constructing the anisotropic Graph Laplacian $L = D - W$.

### C. Slow Epigenetic Chromatin Remodeling on *SLC1A2*
Systemic allostatic load ($\Omega_{\text{ext}}$) crosses the blood-brain barrier via neurovascular filtration lag $\tau_{\text{BBB}} = 45.0\text{ days}$. Progressive glucocorticoid-induced H3K27me3 trimethylation and CpG hypermethylation at the *SLC1A2* promoter govern transcriptionally permissive euchromatin accessibility $E_i(t) \in [1.0 - \Delta_{\max}, 1.0]$ ($\Delta_{\max} = 0.50$, $k_{\text{meth}} = 0.0045\text{ day}^{-1}$, $k_{\text{demeth}} = 0.0060\text{ day}^{-1}$):

$$\frac{dE_i}{dt} = \left(k_{\text{demeth}} \mathcal{B}_m\right)\left(1.0 - E_i(t)\right) - k_{\text{meth}}\left(\Omega_{\text{eff}}(t) - 1.0\right)\left(E_i(t) - (1.0 - \Delta_{\max})\right)$$

$$\frac{du_i}{dt} = \alpha_0 E_i(t) - \beta_{\text{eff}} u_i(t), \qquad \frac{ds_i}{dt} = \beta_{\text{eff}} u_i(t) - \gamma_{\text{eff}} s_i(t), \qquad V_{\max,i}^*(t) = V_{\max,i}^0 \cdot \operatorname{clip}\!\left(\frac{s_i(t)}{s_0}, \, 0.50, \, 1.0\right)$$

with maturation rate $\beta_{\text{eff}} = 0.25\text{ day}^{-1}$ and functional membrane EAAT2 turnover rate $\gamma_{\text{eff}} = 0.12\text{ day}^{-1}$ ($t_{1/2,\text{EAAT2}} \approx 5.8\text{ days}$).

### D. Dual-Regime Formulation: Senescence Attrition (Regime I) vs. Glycogen Mass Balance (Regime II)
To reconcile late-life mitochondrial senescence with early-onset overload in young adults without phenomenological curve-fitting, the ensemble ($N = 200,000$ total virtual subjects) is evaluated across two complementary regimes:
1. **Regime I (Chronic Senescence-Driven Attrition):** Models emergency astrocytic glycogenolysis as a non-depleting structural buffer ($R_{\text{rec},i}(t) \equiv 1.0$), isolating structural vulnerability governed by polygenic mitochondrial reserve ($\text{RBS}_m$) and chronological senescence ($\Delta S_{\text{mito}}(\text{Age}_m)$, declining at $k_{\text{sen}} = 0.0045\text{ year}^{-1}$ after $\text{Age}_{\text{peak}} = 30.0\text{ years}$).
2. **Regime II (Dynamic Allostatic Overload & Glycogen Mass-Balance Saturation):** Enforces physical substrate mass conservation on intracellular astrocytic glycogen stores $R_{\text{rec},i}(t) \in [0, 1]$ ($C_{\text{glyc}} = \sigma_{\text{stoich}}A_0 = 6.0\text{ mM ATP}\cdot\text{day}$, resynthesis rate $k_{\text{syn}} = \gamma_{\text{eff}} = 0.12\text{ day}^{-1}$):

$$\frac{dR_{\text{rec},i}}{dt} = \left(k_{\text{syn}} \mathcal{B}_m\right) f_{\text{auto}}(A_i(t)) \left(1.0 - R_{\text{rec},i}(t)\right) - \frac{\rho_{\text{glyc}} R_{\text{rec},i}(t) \chi_{\text{crisis}}(A_i) \left(1.0 + w_i(\Omega_{\text{eff}}(t) - 1.0)\right)}{C_{\text{glyc}}}$$

---

## 2. Empirical Summary Benchmark ($N = 200,000$ Virtual Subjects)

|Polygenic Reserve Tier ($\text{NASEM 2023}$)|Mean $\mu_{\text{RBS}}$|Sub-Cohort ($\Omega_{\text{ext}} > 3.0$)|Regime I Collapse Rate|Regime I Median $t_c$ (Days)|Regime II Collapse Rate|Regime II Median $t_c$ (Days)|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
|**Tier IV (Low Reserve)**|`0.9457`|`8,320`|`69.75%`|`106.10`|`85.70%`|`93.10`|
|**Tier III (Moderate-Low)**|`0.9810`|`8,423`|`63.07%`|`118.40`|`81.02%`|`102.20`|
|**Tier II (Moderate-High)**|`1.0109`|`8,153`|`55.86%`|`123.30`|`75.17%`|`108.40`|
|**Tier I (High Reserve)**|`1.0462`|`8,282`|`46.64%`|`135.20`|`67.88%`|`118.90`|
|**Total Cohort ($\Omega_{\text{ext}} \in [1, 4]$)**|`1.0000`|`100,000`|**`21.20%`** (Age `57.03`)|—|**`29.81%`** (Age `55.03`)|—|
|**Young Adults (Age 20–35, $\Omega_{\text{ext}} > 3.2$)**|—|—|**`38.36%`**|—|**`64.66%`**|—|

---

## 3. Modular Pipeline Architecture

* **`src/script01_data_ingestion.py`**: Extracts the 148-node Destrieux cortical parcellation (`aparc.a2009s`), computes callosal-corrected white-matter tract distances and the anisotropic Graph Laplacian ($L$), parameterizes deterministic 148-node OXPHOS $P_{\text{ref},i} \in [6.20, 8.60]\text{ mM}\cdot\text{day}^{-1}$ (AHBA/`abagen`) and EAAT2 $V_{\max,i}^0 \in [3.60, 5.80]\text{ mM}\cdot\text{day}^{-1}$ ($[^{11}\text{C}]\text{ABP688}$ PET/`neuromaps`), and locks cell-type-adjusted polygenic reserve tiers (NASEM 2023).
* **`src/script02_sde_pde_engine.py`**: Executes the 4D spatiotemporal parameter sweep across Ages 20–60 and Tiers I–IV under both **Regime I** and **Regime II**, plus the comparative post-stress remission vs. hysteresis challenge ($\Omega_{\text{ext}} = 3.65$ for Days 0–550 $\to \Omega_{\text{relief}} = 1.20$ for Days 550–1000, demonstrating full remission at `2.949 mM` in Age 25 vs. hysteresis lock-in at `0.130 mM` in Age 60).
* **`src/script03_multigpu_montecarlo.py`**: High-throughput PyTorch CUDA engine (`torch.float64` 64-bit double precision) executing $N = 100,000$ virtual subjects per regime ($N = 200,000$ total trajectories, $10,000$ steps per subject) concurrently across dual NVIDIA Tesla T4 GPUs (`cuda:0` and `cuda:1`).
* **`src/script04_master_visual_engine.py`**: Ingests the precomputed `.npy` and `.csv` vaults to render all 7 publication-ready figures (`Figure_1_Complexity.png` through `Figure_7_Regime_II_Overload_Recovery.png`) at 300 DPI.

---

## 4. Deployment & Reproducibility Protocol

### A. Precomputed 200,000-Subject Datasets & Empirical Vaults
To inspect or re-render the exact figures and statistical tables without re-running the dual-GPU Monte Carlo integration, the complete precomputed simulation outputs and configuration arrays are publicly archived at:
* **Zenodo Archive (Configuration & Empirical Parcellation Vault):** `https://doi.org/10.5281/zenodo.20593280`
* **Kaggle Dataset (Full 200,000-Subject Monte Carlo Matrices & CSVs):** `https://www.kaggle.com/datasets/cefiyana/isb-monte-carlo-200k-subject`
* **SSRN Preprint Manuscript:** `https://ssrn.com/abstract=6913658`

### B. Sequential Pipeline Execution

```bash
# 1. Clone the repository
git clone [https://github.com/cefiyana-clover/ISB_Stochastic_Framework.git](https://github.com/cefiyana-clover/ISB_Stochastic_Framework.git)
cd ISB_Stochastic_Framework

# 2. Install validated dependencies
pip install -r requirements.txt

# 3. Execute the 4-stage First-Principles SDE-PDE Pipeline
python src/script01_data_ingestion.py
python src/script02_sde_pde_engine.py
python src/script03_multigpu_montecarlo.py
python src/script04_master_visual_engine.py
```
## 5. Dual-Licensing Structure
​To ensure complete algorithmic transparency while protecting theoretical intellectual property, this project operates under a dual-licensing framework:

* **Computational Source Code (src/*.py):** Distributed under the GNU General Public License v3.0 (GPL-3.0). Free for academic use, verification, and modification under copyleft terms.

​* **Theoretical Framework, Equations & Manuscript:** Distributed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

* **​Author & Correspondence:**
Cefiyana — Independent Interdisciplinary Researcher, Indonesia
Email: cefiyana@fiveleafclover.org | ORCID iD: 0009-0008-4324-9515
