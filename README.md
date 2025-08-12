# Multi-Region Traffic Control with SUMO

This repository implements a **Multi-Scale Joint Control (MSJC)** framework for **perimeter control** and **route guidance** in a multi-region traffic network, integrated with [**SUMO**](https://www.eclipse.org/sumo/) (Simulation of Urban MObility).

## 🚦 Control Strategies

We provide five control strategies for comparison:

| Code       | Description |
|------------|-------------|
| **MSJC**   | Proposed **multi-scale joint control** integrating perimeter control and route guidance |
| **MSPC**   | Multi-scale perimeter control without route guidance |
| **MSPC-LR**| Multi-scale perimeter control with **logit-based** route guidance |
| **MP**     | Backpressure control for boundary intersections without route guidance |
| **MP-LR**  | Backpressure control for boundary intersections with **logit-based** route guidance |

## 📂 Repository Structure
data/         # SUMO simulation networks, demand, and configuration files
partition/    # Network partitioning schemes and MFD functions
controllers/  # Upper- and lower-level control algorithm implementations
utils/        # Helper functions for simulation, logging, and result processing

## 📜 Related Paper

> **Peng, Xianyue**, Wang, Hao, & Zhang, Michael.  
> *A Multi-Scale Perimeter Control and Route Guidance System for Large-Scale Road Networks.*  
> SSRN, 2023. [https://ssrn.com/abstract=4502092](https://ssrn.com/abstract=4502092)  

**BibTeX:**
bibtex
@article{peng2023multiscale,
  title={A Multi-Scale Perimeter Control and Route Guidance System for Large-Scale Road Networks},
  author={Peng, Xianyue and Wang, Hao and Zhang, Michael},
  journal={SSRN Electronic Journal},
  year={2023},
  doi={10.2139/ssrn.4502092}
}

More works by the author can be found on Google Scholar: https://scholar.google.com/citations?user=o8ghKIgAAAAJ&hl=en&oi=ao
