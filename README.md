# Laplace Operator on Metric Graphs

This repository contains the code used in "A continuum limit for dense spatial networks".

---

## 🧠 Overview

This project implements and analyzes solutions to the 1D Laplace/Helmholtz equation on various types of **metric graphs**. It includes:

- A custom metric graph construction framework
- Solver for the Laplace operator on graphs
- Reproduction scripts for all paper figures

---

## 📦 Repository Structure

- spiderweb: run 'python spiderweb.py' to reproduce eigenmode convergence plots
- homogenization_coefficient: contains notebooks for reproducing convergence plots for the homogenization coefficient and deviatorics
- random graphs: contains notebooks for reproducing eigenmode convergence plots for the Delaunay triangulation, RGG, and aperiodic monotile
- random_inhomogeneous: contains notebooks for solving the inhomogeneous problem in the continuum using Dedalus (https://dedalus-project.org/) and on the graph
