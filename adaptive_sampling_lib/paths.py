"""Where results, data and model weights live.  Every location can be set with an environment variable; the defaults
are the lab's shared storage.

  ADAPTIVE_SAMPLING_RESULTS     result JSONs (one sub-folder per experiment)
  ADAPTIVE_SAMPLING_DATA_CACHE  downloaded datasets (OpenML, California housing) and the built MIMIC-IV cohort
  ADAPTIVE_SAMPLING_EICU_DIR    eICU files (credentialed data; not distributed with this code)
  ADAPTIVE_SAMPLING_MIMIC_DIR   raw MIMIC-IV v3.1 tables (credentialed data; not distributed with this code)
  ADAPTIVE_SAMPLING_MODEL_CACHE Kumo-Tabular weights (hf_hub/) and the structured-data-models package (sdm_pkgs/)
  TABPFN_MODEL_CACHE_DIR        TabPFN checkpoints (the tabpfn package's own variable)
"""
import os

_SHARED = "/shared/share_mala/daksh/adaptive_sampling_experiments"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("ADAPTIVE_SAMPLING_RESULTS", os.path.join(_SHARED, "results"))
DATA_CACHE = os.environ.get("ADAPTIVE_SAMPLING_DATA_CACHE", os.path.join(_SHARED, "data_cache"))
EICU_DIR = os.environ.get("ADAPTIVE_SAMPLING_EICU_DIR", "/shared/share_mala/data/eicu_train_test")
MIMIC_DIR = os.environ.get("ADAPTIVE_SAMPLING_MIMIC_DIR", "/shared/share_mala/daksh/mimiciv/physionet.org/files/mimiciv/3.1")
MODEL_CACHE = os.environ.get("ADAPTIVE_SAMPLING_MODEL_CACHE", "/shared/share_mala/daksh/model_cache")
TABPFN_MODELS = os.environ.get("TABPFN_MODEL_CACHE_DIR", os.path.join(_SHARED, "tabpfn_models"))


def results(*parts):
    """Path under the results root, e.g. results("gp_high_dim")."""
    return os.path.join(RESULTS_ROOT, *parts)
