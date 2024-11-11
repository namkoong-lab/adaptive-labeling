# **Adaptive Labeling for Efficient Out-of-Distribution Model Evaluation**

## Introduction

Supervised data suffers severe selection bias when labels are expensive. We formulate a MDP over posterior beliefs on model performance and solve it with pathwise policy gradients computed through an auto-differentiable pipeline. The paper is available [here](https://openreview.net/pdf?id=uuQQwrjMzb).

**Key Features:**
- Adaptive Labeling - MDPs with combinatorial action space
- Uncertainty Quantification - Gaussian Processes, Deep Learning based UQ methodologies (Ensembles, Ensemble+, ENNs)
- Policy Parametrization through K-subset sampling
- Policy Gradients through Autodiff - Smoothed Differentiable Pipeline 

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Creating the Environment](#creating-the-environment)
4. [Running the Project](#running-the-project)
5. [Testing](#testing)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Structure

```plaintext
Project_Name/
│
├── Main/                    # Source code for the project
│   ├── gp_experiments
│        ├── gp_pipeline_regression
             ├── run_pipeline_long_horizon.py
             ├── run_pipeline_pg_long_horizon.py
             ├── run_pipeline_active_learning_long_horizon.py
             └── .... 
              
│        ├── gp_pipeline_regression_real_data
             ├── run_pipeline_long_horizon.py
             ├── run_pipeline_pg_long_horizon.py
             ├── run_pipeline_active_learning_long_horizon.py
             └── .... 
│   └── ensemble_plus_experiments
│    
│
├── src/                   # Source code for ongoing research (under development)
│   ├── autodiff           # Autodiff (Smoothed-Differentiable) pipeline development - different UQ methodologies, integration with baselines
│        ├── gp
│        ├── ensemble_plus
│        ├── enn
│        ├── deprecated    # Deprecated code
│    ├── baselines          # REINFORCE based policy gradient pipeline development
│    └──  notebooks          # Notebooks for unit tests, testing individual components of the pipeline
│
├── requirements.txt        # List of dependencies
└──  README.md               # Project documentation
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/username/repository_name.git
   cd repository_name
   ```

2. **Install dependencies:**

   Ensure you have Python 3.8 or later installed.

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
conda create -n project_env python=3.8
conda activate project_env
pip install -r requirements.txt
```

---

## Running the Project

After setting up the environment, you can run the project using:

```bash
python src/main.py
```

**Note:** Replace `src/main.py` with the appropriate entry point of the project if different.

---

## Testing

To run tests, use `pytest` or any other testing framework specified in `requirements.txt`.

```bash
pytest tests/
```

This will run all tests in the `tests` directory.

---

## Usage

Once installed and set up, the project can be used as follows:

1. **Data Preprocessing**: Run `src/data_processing.py` to clean and prepare data.
2. **Model Training**: Run `src/model_training.py` to train the model.
3. **Evaluation**: Run `src/evaluation.py` for evaluation and metrics.

Modify the parameters in `config.py` as needed for custom settings.

---




