"""Where results, data and model weights live.  Every location can be set with an environment variable; by default
everything is kept in folders inside the repository (ignored by git).

  ADAPTIVE_SAMPLING_RESULTS     result JSONs (one sub-folder per experiment)            default <repo>/results
  ADAPTIVE_SAMPLING_DATA_CACHE  downloaded datasets (OpenML, California housing) and     default <repo>/data_cache
                                the built MIMIC-IV cohort
  ADAPTIVE_SAMPLING_EICU_DIR    eICU files (credentialed data; not distributed)          default <repo>/data/eicu
  ADAPTIVE_SAMPLING_MIMIC_DIR   raw MIMIC-IV v3.1 tables (credentialed data; not         default <repo>/data/mimiciv/3.1
                                distributed)
  ADAPTIVE_SAMPLING_MODEL_CACHE Kumo-Tabular weights (hf_hub/) and the                   default <repo>/model_cache
                                structured-data-models package (sdm_pkgs/)
  TABPFN_MODEL_CACHE_DIR        TabPFN checkpoints (the tabpfn package's own variable)  default <model cache>/tabpfn
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("ADAPTIVE_SAMPLING_RESULTS", os.path.join(REPO_ROOT, "results"))
DATA_CACHE = os.environ.get("ADAPTIVE_SAMPLING_DATA_CACHE", os.path.join(REPO_ROOT, "data_cache"))
EICU_DIR = os.environ.get("ADAPTIVE_SAMPLING_EICU_DIR", os.path.join(REPO_ROOT, "data", "eicu"))
MIMIC_DIR = os.environ.get("ADAPTIVE_SAMPLING_MIMIC_DIR", os.path.join(REPO_ROOT, "data", "mimiciv", "3.1"))
MODEL_CACHE = os.environ.get("ADAPTIVE_SAMPLING_MODEL_CACHE", os.path.join(REPO_ROOT, "model_cache"))
TABPFN_MODELS = os.environ.get("TABPFN_MODEL_CACHE_DIR", os.path.join(MODEL_CACHE, "tabpfn"))


def results(*parts):
    """Path under the results root, e.g. results("gp_high_dim")."""
    return os.path.join(RESULTS_ROOT, *parts)
