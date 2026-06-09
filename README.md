
# Bioenergetic Stability Index (ISB): A Multi-Scale SDE-PDE Framework

**Status:** Production-Ready | **Validation:** Empirical In-Vivo/In-Vitro Anchors | **Architecture:** OS-Agnostic  

This repository contains the computational framework for the Bioenergetic Stability Index (ISB). It executes a multi-scale spatiotemporal simulation linking cellular transcriptomics, thermodynamic limits, and macroscopic topological phase transitions within a 6-Dimensional Manifold framework.

## 1. Biophysical Architecture & Micro-Macro Bridge

The core engine models astrocytic-neuronal metabolic coupling through a system of Stochastic Differential Equations (SDE) and Partial Differential Equations (PDE). The architecture enforces strict thermodynamic boundaries (preventing negative energy state anomalies) and strictly dissipative diffusion mechanics.

### A. Stochastic Langevin Noise (SDE)
Astrocytic ATP depletion is modeled using the Euler-Maruyama integration method, incorporating a purely stochastic Wiener process to simulate thermodynamic variance without artificial dampening.

$$d[ATP] = \left( P_{basal} - \frac{\Omega_{ext} \cdot \xi \cdot [Glu]}{K_{m,ATP} + [Glu]} - k_{decay}[ATP] + \rho \cdot \mathcal{H}([ATP] - \theta) \right) dt + \sigma [ATP] dW_t$$

### B. Anisotropic Euclidean Diffusion (PDE)
Glutamatergic excitotoxicity propagates strictly along structural white-matter tracts via a Graph Laplacian matrix ($L$) derived from the Destrieux 2009 anatomical parcellation, obeying Fick's Law of Diffusion.

$$d[Glu] = \left( k_{prod}[ATP] - \frac{V_{max}^* [Glu]}{K_{m,G} + [Glu]} - D_G (L \cdot [Glu]) \right) dt$$

### C. Dynamic Epigenetic Drift (RNA Velocity Kinetics)
The framework bridges micro-scale genetic transcription to macro-scale topological collapse by modeling the delay in Excitatory Amino Acid Transporter (EAAT) degradation.

$$\frac{du}{dt} = \alpha_0 e^{-\lambda (\Omega_{ext} - 1)} - \beta u$$
$$\frac{ds}{dt} = \beta u - \gamma s$$

### D. Systemic Parameterization
All variables are anchored to empirical clinical limits:
* **Topology:** Destrieux Atlas (L2 DTI Proxy)
* **Transcriptomics:** Allen Human Brain Atlas (AHBA)
* **Receptor Density:** mGluR5 PET Neuromaps
* **Genetic Variance:** gnomAD mtDNA-CN distributions

## 2. Execution Pipeline

The framework is strictly modularized to maintain procedural transparency and resource efficiency.

* **`src/script01_data_ingestion.py`**: Deterministic extraction of multimodal empirical datasets. Constructs foundational matrices with synthetic fallback mechanisms to guarantee pipeline stability.
* **`src/script02_sde_integration.py`**: Executes the SDE-PDE double steady-state calibration to ensure baseline homeostatic strictness before allostatic load injection.
* **`src/script03_megasimulation.py`**: Parallelized Monte Carlo engine for large-scale cohort execution ($N=100,000$). Optimized via `joblib` for HPC architectures.
* **`src/script04_visual_rendering.py`**: Ingests `.npy` spatiotemporal matrices to render deterministic publication-ready topological heatmaps, phase space trajectories, and survival probability estimates.

## 3. Deployment Protocol

Execute the pipeline sequentially to preserve data dependencies. The scripts are OS-Agnostic and automatically anchor the operational directory relative to the execution path.

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/ISB-Stochastic-Framework.git
cd ISB-Stochastic-Framework

# 2. Install empirically validated dependencies
pip install -r requirements.txt

# 3. Execute the SDE-PDE Pipeline
python src/script01_data_ingestion.py
python src/script02_sde_integration.py
python src/script03_megasimulation.py
python src/script04_visual_rendering.py

```
## 4. Dual-License Architecture
To ensure operational transparency while protecting theoretical intellectual property, this project adopts a dual-licensing structure:
 * **Computational Source Code:** Distributed under the **GNU General Public License v3.0 (GPL-3.0)**. The SDE-PDE execution framework remains open, free, and mechanically transparent for the global scientific community.
 * **Theoretical Framework & Manuscript:** The underlying mathematical formulation, architectural concepts, and associated manuscript texts are distributed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. Academic usage and integration require strict empirical referencing to the original Bioenergetic Stability Index (ISB) architecture.
```

**Academic Correspondence:** <cefiyana@fiveleafclover.org>
