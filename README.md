# Multi-Region Traffic Control with SUMO
This repository implements a multi-scale joint control (MSJC) framework for perimeter control and route guidance in a multi-region network, integrated with SUMO (Simulation of Urban MObility).

Control Strategies:
1. MSJC: proposed multi-scale joint control of perimeter control and route guidance
2. MSPC: multi-scale perimeter control without route guidance
3. MSPC-LR: multi-scale perimeter control with a logit-based route guidance strategy
4. MP: backpressure control for boundary intersections without route guidance;
5. MP-LR: backpressure control for boundary intersection with logit-based route guidance strategy

The SUMO simulation files are located in the data directory.

The network partitioning scheme and MFD functions are located in the partition directory.

Related Paper: 

Peng, X., Wang, H., & Zhang, M. A Multi-Scale Perimeter Control and Route Guidance System for Large-Scale Road Networks. Available at SSRN 4502092.

Abstract: 
Perimeter control and route guidance are effective ways to reduce traffic congestion and improve traffic efficiency by controlling the spatial and temporal traffic distribution on the network. This paper presents a multi-scale joint perimeter control and route guidance (MSJC) framework for controlling traffic in large-scale networks. The network is first partitioned into several subnetworks (regions) with traffic in each region governed by its macroscopic fundamental diagram (MFD), which forms the macroscale network (upper level). Each subnetwork, comprised of actual road links and signalized intersections, forms the microscale network (lower level). At the upper level, a joint perimeter control and route guidance model solves the region-based inflow rate and hyper-path flows to control the accumulation of each region and thus maximize the throughput of each region. At the lower level, a perimeter control strategy integrated with a backpressure policy determines the optimal signal phases of the intersections at the regional boundary. At the same time, a route choice model for vehicles is constructed to meet hyper-path flows and ensure the intra-region homogeneity of traffic density. The case study results demonstrate that the proposed MSJC outperforms other benchmarks in regulating regional accumulation, thereby improving throughput.

Other Papers by the Author:

Wang, H., & Peng, X. (2022). Coordinated control model for oversaturated arterial intersections. IEEE Transactions on Intelligent Transportation Systems, 23(12), 24157-24175.

Peng, X., & Wang, H. (2023). Network-wide coordinated control based on space-time trajectories. IEEE Intelligent Transportation Systems Magazine, 15(4), 72-85.

Peng, X., & Wang, H. (2023). Coordinated control model for arterials with asymmetric traffic. Journal of Intelligent Transportation Systems, 27(6), 752-768.

Peng, X., & Wang, H. (2024). Capturing Spatial-Temporal Traffic Patterns: A Dynamic Partitioning Strategy for Heterogeneous Traffic Networks. IEEE Access.

Peng, X., et al. (2023). Joint Optimization of Traffic Signal Control and Vehicle Routing in Signalized Road Networks using Multi-Agent Deep Reinforcement Learning. arXiv preprint arXiv:2310.10856.

Peng, X., et al. (2023). Combat Urban Congestion via Collaboration: Heterogeneous GNN-based MARL for Coordinated Platooning and Traffic Signal Control. arXiv preprint arXiv:2310.10948.
