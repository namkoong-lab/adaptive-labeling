# **Adaptive Labeling for Efficient Out-of-Distribution Model Evaluation**

## Introduction

Supervised data suffers severe selection bias when labels are expensive. We formulate a MDP over posterior beliefs on model performance and solve it with pathwise policy gradients computed through an auto-differentiable pipeline. The paper is available [here](https://openreview.net/pdf?id=uuQQwrjMzb).

**Key Features:**
- Adaptive Labeling - MDPs with combinatorial action space
- Uncertainty Quantification - Gaussian Processes, Deep Learning based UQ methodologies (Ensembles, Ensemble+, ENNs)
- Policy parametrization through K-subset sampling
- Policy gradients through Autodiff - Smoothed differentiable pipeline 

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Creating the Environment](#creating-the-environment)
4. [Running the Project](#running-the-project)

---

## Project Structure

```plaintext
Project_Name/
│
├── Main/                    # Source code for the project
│   ├── gp_experiments
│   │    ├── gp_pipeline_regression
│   │    │     ├── run_pipeline_long_horizon.py
│   │    │     ├── run_pipeline_pg_long_horizon.py
│   │    │     ├── run_pipeline_active_learning_long_horizon.py
│   │    │     └── .... 
│   │    │       
│   │    └── gp_pipeline_regression_real_data
│   │          ├── run_pipeline_long_horizon.py
│   │          ├── run_pipeline_pg_long_horizon.py
│   │          ├── run_pipeline_active_learning_long_horizon.py
│   │          └── .... 
│   └── ensemble_plus_experiments
│          ├── ensemble_plus_pipeline_regression
│          │       ├── run_enn_pipeline_1a.py
│          │       ├── run_pipeline_pg_ensemble_plus_long_horizon.py
│          │       └── ....
│          └── ensemble_plus_pipeline_regression_active_learning
│                  ├── run_enn_pipeline_1a_active_learning.py
│                  └── ....
│    
│
├── src/                   # Source code for ongoing research (under development)
│   ├── autodiff           # Autodiff (Smoothed-Differentiable) pipeline development - different UQ methodologies, integration with baselines
│   │     ├── gp
│   │     ├── ensemble_plus
│   │     ├── enn
│   │     └── deprecated    # Deprecated code
│   ├── baselines          # REINFORCE based policy gradient pipeline development
│   └──  notebooks          # Notebooks for unit tests, testing individual components of the pipeline
│
├── requirements.txt        # List of dependencies
└──  README.md               # Project documentation
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/namkoong-lab/adaptive-labeling.git
   cd adaptive-labeling
   ```

2. **Install dependencies:**

   We use Python 3.10.13 for our experiments.

   ```bash
   pip install -r requirements.txt
   ```

---

## Creating the Environment

To ensure a consistent and isolated environment, you can create a virtual environment using `venv` or `conda`.

### Using `venv`

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

### Using `conda`

```bash
conda create -n project_env python=3.10
conda activate project_env
pip install -r requirements.txt
```

---

## Running the Project

1. We currently use weights and biases ([Link](https://wandb.ai/site/)) to track our project and our code is integrated with wandb. See ([Link](https://wandb.ai/site/)) for setting up an account.
2. Accordingly one might need to edit files for including their own ``ENTITY'' name on wandb.

         For example - In line 288 of "Main/gp_experiments/gp_pipeline_regression/run_pipeline_long_horizon.py" - put your own entity name
   
3. After setting up the environment, one can run various pipelines (AUTODIFF, REINFORCE, ACTIVE LEARNING) of the project using following command line (similar commnad line for other pipelines) :

```bash
python Main/gp_experiments/gp_pipeline_regression/run_pipeline_long_horizon.py --config_file_path Main/gp_experiments/gp_pipeline_regression/config_sweep_0.json --project_name gp_adaptive_sampling_final_run
```


### Citation

```bib
@inproceedings{
    mittal2024adaptive,
    title={Adaptive Labeling for Efficient Out-of-distribution Model Evaluation},
    author={Daksh Mittal and Yuanzhe Ma and Shalmali Joshi and Hongseok Namkoong},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=uuQQwrjMzb}
}
```







