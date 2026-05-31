# Object-Oriented Dwelling Simulator

A Python-based simulation framework for modeling and analyzing dwelling systems using object-oriented design principles. This project is designed for experimentation, simulation reproducibility, and structured analysis of dwelling-related scenarios.

---

## 📦 Requirements

* Python>= 3.10 
* Conda (recommended for environment management)

---

## ⚙️ Environment Setup

### 1. Create a virtual environment

```bash
conda create -n OODS python=3.9.21
```

### 2. Activate the environment

```bash
conda activate OODS
```

### 3. Clone the repository

```bash
git clone https://github.com/yaju1995/Object-Oriented-Dwelling-Simulator-Public.git
```

### 4. Navigate to the project directory

```bash
cd Object-Oriented-Dwelling-Simulator-Public
```

### 5. Install dependencies

```bash
pip install -e .
```

---

## 🚀 Running the Simulation

### Option 1: Using Jupyter Notebook

1. Navigate to the simulation folder:

   ```bash
   cd run
   ```
2. Launch Jupyter:

   ```bash
   jupyter notebook
   ```
3. Open and run:

   ```
   Dwelling Simulation.ipynb
   ```

---

### Option 2: HEMS Training Script

Run the simulation directly:

```bash
HEMS_Dwelling_train.py
```

---

### Option 3: HEMS Test Script

Run the simulation directly:

```bash
HEMS_Dwelling_test.py
```



---

## 🧱 Project Structure

```
Object-Oriented-Dwelling-Simulator-Public/
│
├── run/              # Simulation scripts and notebooks
├── Results/        # Analysis and visualization notebooks
├── src/                    # Core object-oriented implementation
├── pyproject.toml        # Dependencies (if applicable)
└── Readme.md               # Project documentation
```

---

## 🧠 Overview

This project leverages object-oriented programming (OOP) to:

* Model dwelling components as modular objects
* Simulate interactions between entities
* Enable reproducible experimental setups
* Provide a flexible framework for future extensions

---

## 📌 Notes

* Ensure all dependencies are installed correctly before running simulations.
* For best reproducibility, use the specified Python version.
* Modify configuration parameters inside scripts/notebooks to explore different scenarios.


