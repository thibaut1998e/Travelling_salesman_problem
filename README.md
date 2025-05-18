# Travelling Salesman Problem (TSP)

This project explores different algorithms to solve the **Travelling Salesman Problem (TSP)**, a classic optimization problem in computer science and operations research. It was done in the context of an assignement in 02/2020 at university of Oslo. This readme briefly summarizes the work, for more details see *Thibaut_ELOY_First_Mandatory_Assignement.pdf*

## 🧭 What is the Travelling Salesman Problem?

The **Travelling Salesman Problem (TSP)** is defined as follows:

> Given a list of cities and the distances between each pair of them, what is the shortest possible route that visits each city exactly once and returns to the origin city?

TSP is **NP-hard**, meaning there is no known algorithm that can solve all instances of the problem efficiently (i.e., in polynomial time). As the number of cities increases, the number of possible routes grows factorially, making brute-force approaches computationally impractical for larger instances.

---

## 🧪 Methods Implemented

This project includes several algorithms to approach the TSP, ranging from exhaustive methods to evolutionary approaches.

### 🔍 Exhaustive Search

This brute-force method evaluates **all possible permutations** of city routes to find the optimal one. While it guarantees the best solution, it is only feasible for a **very small number of cities** due to its **factorial time complexity (O(n!))**.

### ⛰️ Hill Climbing

A **local search** algorithm that starts with a random solution and iteratively makes small changes to improve it. It’s faster than exhaustive search but may get stuck in **local optima** since it only accepts changes that improve the current solution.

### 🧬 Genetic Algorithm

Inspired by the process of natural selection, the **genetic algorithm** works by maintaining a population of candidate solutions. It applies **selection**, **crossover**, and **mutation** to evolve better routes over generations. It provides good approximate solutions and handles large search spaces more efficiently than local methods.

### 🔁 Hybrid Algorithms

This project also explores **hybrid evolutionary strategies** that combine genetic algorithms with local search techniques:

- **Lamarckian Evolution**: Individuals improve through local optimization (e.g., hill climbing), and the resulting improved genotype is inherited by offspring. This assumes that acquired characteristics are passed on, accelerating convergence.
  
- **Baldwinian Evolution**: Similar to Lamarckian, individuals are locally optimized, but only the **fitness** is updated—not the genotype. This retains diversity in the population while still guiding selection toward better solutions.
