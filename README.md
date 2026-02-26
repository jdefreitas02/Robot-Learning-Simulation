# Robot Learning: Policy Distillation & Model-Based RL

This repository contains a hybrid Robot Learning architecture designed to navigate a complex, high-resistance environment characterised by severe state-dependent rotational dynamics and strict financial/sample budget constraints.

## Overview

Traditional Model-Based Reinforcement Learning (MBRL) with a Model Predictive Control (MPC) planner often struggles in environments with compounding model errors and state-dependent noise, leading to volatile test-time performance. Pure Behavioural Cloning (BC) fails due to covariate shift when the robot drifts off the expert path.

This project solves both issues by implementing a **Policy Distillation** framework. It uses a computationally heavy, mathematically optimal planner (CEM) as a "Teacher" to generate a vast, robust dataset of synthetic recovery trajectories. A fast, reactive neural network (the "Student") is then trained on this data to execute the final policy.

## Methodology

Our pipeline consists of five core components:

1. **Forward Dynamics Model:** We collect a minimal set of expert demonstrations and train a forward dynamics neural network (`128x128`) to approximate the environment's transition MDP.

2. **The CEM "Teacher" (Algorithmic Expert):** We use the Cross-Entropy Method (CEM) to track the expert demonstrations. To maximise sample efficiency without incurring environment reset penalties, the robot explores strictly along the bounds of the demonstration path, alternating in forward and backward passes.

3. **DAgger-Style Synthetic Data Generation:** During the CEM teacher's rollouts, we inject Gaussian noise into the executed actions. This forces the robot slightly off the optimal path into unvisited "danger" states. We then record the un-noised, mathematically optimal CEM action required to recover.

4. **Progress Filtering:** To prevent the integration of sub-optimal data (e.g., getting stuck in heavy "mud"), strict heuristic progress filtering is applied. Synthetic trajectories where the teacher stalls are discarded.

5. **Behavioural Cloning "Student":** A purely reactive BC policy is trained on the synthetically generated dataset alongside the original demonstrations. During testing, the robot executes this policy in $O(1)$ time, implicitly counteracting state-dependent dynamics and bypassing MPC volatility.

6. **Smart Fallback Recovery:** If the BC model ever encounters a completely alien state and gets stuck during testing, the system temporarily wakes up the MBRL CEM planner to simulate an optimal escape route before handing control back to the BC model.

## How to Run

To execute the training pipeline and evaluate the robot's performance in the testing environment, simply run the main script from your terminal:

```bash
python robot-learning.py
