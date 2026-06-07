# Bioenergetic Stability Index (ISB): A Multi-Scale SDE-PDE Framework
**Status:** Production-Ready | **Validation:** Empirical In-Vivo/In-Vitro Anchors | **Architecture:** OS-Agnostic
This repository contains the computational framework for the Bioenergetic Stability Index (ISB). It executes a multi-scale spatiotemporal simulation linking cellular transcriptomics, thermodynamic limits, and macroscopic topological phase transitions within a 6-Dimensional Manifold framework.
## 1. Biophysical Architecture & Micro-Macro Bridge
The core engine models astrocytic-neuronal metabolic coupling through a system of Stochastic Differential Equations (SDE) and Partial Differential Equations (PDE). The architecture enforces strict thermodynamic boundaries (preventing negative energy state anomalies) and strictly dissipative diffusion mechanics.
### A. Stochastic Langevin Noise (SDE)
Astrocytic ATP depletion is modeled using the Euler-Maruyama integration method, incorporating a purely stochastic Wiener process to simulate thermodynamic variance without artificial dampening.
### B. Anisotropic Euclidean Diffusion (PDE)
Glutamatergic excitotoxicity propagates strictly along structural white-matter tracts via a Graph Laplacian matrix (L) derived from the Destrieux 2009 anatomical parcellation, obeying Fick's Law of Diffusion.
### C. Dynamic Epigenetic Drift (RNA Velocity Kinetics)
The framework bridges micro-scale genetic transcription to macro-scale topological collapse by modeling the delay in Excitatory Amino Acid Transporter (EAAT) degradation.
### D. Systemic Parameterization
All variables are anchored to empirical clinical limits:
 * **Topology:** Destrieux Atlas (L2 DTI Proxy)
 * **Transcriptomics:** Allen Human Brain Atlas (AHBA)
 * **Receptor Density:** mGluR5 PET Neuromaps
 * **Genetic Variance:** gnomAD mtDNA-CN distributions
## 2. Execution Pipeline
The framework is strictly modularized to maintain procedural transparency and resource efficiency.
 * **src/script01_data_ingestion.py**: Deterministic extraction of multimodal empirical datasets. Constructs foundational matrices with synthetic fallback mechanisms to guarantee pipeline stability.
 * **src/script02_sde_integration.py**: Executes the SDE-PDE double steady-state calibration to ensure baseline homeostatic strictness before allostatic load injection.
 * **src/script03_megasimulation.py**: Parallelized Monte Carlo engine for large-scale cohort execution (N=100,000). Optimized via joblib for HPC architectures.
 * **src/script04_visual_rendering.py**: Ingests .npy spatiotemporal matrices to render deterministic publication-ready topological heatmaps, phase space trajectories, and survival probability estimates.
## 3. Deployment Protocol
Execute the pipeline sequentially to preserve data dependencies. The scripts are OS-Agnostic and automatically anchor the operational directory relative to the execution path.
```bash
# 1. Clone the repository
git clone https://github.com/[YOUR-USERNAME]/ISB-Stochastic-Framework.git
cd ISB-Stochastic-Framework

# 2. Install empirically validated dependencies
pip install -r requirements.txt

# 3. Execute the SDE-PDE Pipeline
python src/script01_data_ingestion.py
python src/script02_sde_integration.py
python src/script03_megasimulation.py
python src/script04_visual_rendering.py

```
## 4. License
This repository and its foundational mechanistic framework are distributed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. Academic usage requires adherence to strict empirical referencing standards.
