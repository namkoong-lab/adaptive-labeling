#!/usr/bin/env python
"""
Script to train and save a predictor network once for real data experiments.
This predictor can then be loaded by all pipeline scripts (pg, non-pg, active learning)
to ensure consistent predictor across all experiments and seeds.

Usage:
    # For CSV data:
    python train_and_save_predictor.py --csv_directory /path/to/data/ --output_path predictor_trained.pt

    # For synthetic data:
    python train_and_save_predictor.py --config_file_path config_sweep.json --output_path predictor_trained.pt

The predictor will be trained on the training data.
"""

import numpy as np
import pandas as pd
import torch
import gpytorch
import argparse
import json
import wandb

import warnings
warnings.filterwarnings('ignore')

import polyadic_sampler_new as polyadic_sampler
from predictor_network import train_predictor, save_predictor


def main():
    # Initialize wandb in offline mode (required by polyadic_sampler)
    wandb.init(mode="offline", project="predictor_training")
    parser = argparse.ArgumentParser(description="Train and save predictor network")
    parser.add_argument("--config_file_path", type=str,
                        help="Path to the JSON config file (for synthetic data)",
                        default=None)
    parser.add_argument("--csv_directory", type=str,
                        help="Path to CSV data directory (for real data)",
                        default=None)
    parser.add_argument("--output_path", type=str,
                        help="Path to save the trained predictor",
                        default='predictor_trained.pt')
    parser.add_argument("--seed_predictor", type=int,
                        help="Seed for predictor training (separate from dataset seed)",
                        default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Predictor training seed: {args.seed_predictor}")

    # Load data
    if args.csv_directory is not None:
        # Load from CSV
        print(f"Loading data from CSV directory: {args.csv_directory}")

        train_x = pd.read_csv(args.csv_directory + "train_x.csv")
        numpy_array_train_x = train_x.to_numpy()
        train_x = torch.tensor(numpy_array_train_x, dtype=torch.float32)[:, 1:]

        train_y = pd.read_csv(args.csv_directory + "train_y.csv")
        numpy_array_train_y = train_y.to_numpy()
        train_y = (torch.tensor(numpy_array_train_y, dtype=torch.float32)[:, 1]).squeeze()

        train_x = train_x.to(device)
        train_y = train_y.to(device)
        input_dim = train_x.size(1)

    elif args.config_file_path is not None:
        # Load from config (synthetic data generation)
        print(f"Generating synthetic data from config: {args.config_file_path}")
        with open(args.config_file_path, 'r') as config_file:
            config_params = json.load(config_file)

        # Extract parameters from config (take first value from each list)
        params = config_params['parameters']

        def get_value(key, default=None):
            if key in params:
                values = params[key].get('values', [default])
                return values[0] if values else default
            return default

        # Dataset parameters
        seed_dataset = get_value('seed_dataset', 0)
        no_train_points = get_value('no_train_points', 100)
        no_test_points = get_value('no_test_points', 285)
        no_pool_points = get_value('no_pool_points', 500)
        dataset_model_name = get_value('dataset_model_name', 'GP')
        no_anchor_points = get_value('no_anchor_points', 51)
        input_dim = get_value('input_dim', 1)
        stdev_scale = get_value('stdev_scale', 0.5)
        stdev_pool_scale = get_value('stdev_pool_scale', 0.5)
        scaling_factor = get_value('scaling_factor', 1.0)
        scale_by_input_dim = get_value('scale_by_input_dim', True)
        gp_model_dataset_generation = get_value('gp_model_dataset_generation', 'specify_own')
        stdev_blr_w = get_value('stdev_blr_w', 0.1)
        stdev_blr_noise = get_value('stdev_blr_noise', 0.01)
        logits = get_value('logits', None)
        if_logits = get_value('if_logits', True)
        if_logits_only_pool = get_value('if_logits_only_pool', False)
        plot_folder = get_value('plot_folder', 'none')
        direct_tensors_bool = get_value('direct_tensors_bool', True)

        # Dataset GP parameters
        dataset_mean_constant = get_value('dataset_mean_constant', 0.0)
        dataset_length_scale = get_value('dataset_length_scale', 1.0)
        dataset_output_scale = get_value('dataset_output_scale', 0.69)
        dataset_noise_std = get_value('dataset_noise_std', 0.1)

        # Set seed for data generation (same as in pipeline scripts)
        torch.manual_seed(seed_dataset)
        np.random.seed(seed_dataset)
        if device == "cuda":
            torch.cuda.manual_seed(seed_dataset)
            torch.cuda.manual_seed_all(seed_dataset)

        # Generate data (same logic as in pipeline scripts)
        if direct_tensors_bool:
            if dataset_model_name == "blr":
                polyadic_sampler_cfg = polyadic_sampler.PolyadicSamplerConfig(
                    no_train_points=no_train_points, no_test_points=no_test_points,
                    no_pool_points=no_pool_points, model_name=dataset_model_name,
                    no_anchor_points=no_anchor_points, input_dim=input_dim,
                    stdev_scale=stdev_scale, stdev_pool_scale=stdev_pool_scale,
                    scaling_factor=scaling_factor, scale_by_input_dim=scale_by_input_dim,
                    model=None, stdev_blr_w=stdev_blr_w, stdev_blr_noise=stdev_blr_noise,
                    logits=logits, if_logits=if_logits, if_logits_only_pool=if_logits_only_pool,
                    plot_folder=plot_folder
                )
                train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx = \
                    polyadic_sampler.set_data_parameters_and_generate(polyadic_sampler_cfg)

            elif dataset_model_name == "GP":
                if gp_model_dataset_generation == "use_default":
                    polyadic_sampler_cfg = polyadic_sampler.PolyadicSamplerConfig(
                        no_train_points=no_train_points, no_test_points=no_test_points,
                        no_pool_points=no_pool_points, model_name=dataset_model_name,
                        no_anchor_points=no_anchor_points, input_dim=input_dim,
                        stdev_scale=stdev_scale, stdev_pool_scale=stdev_pool_scale,
                        scaling_factor=scaling_factor, scale_by_input_dim=scale_by_input_dim,
                        model=None, stdev_blr_w=stdev_blr_w, stdev_blr_noise=stdev_blr_noise,
                        logits=logits, if_logits=if_logits, if_logits_only_pool=if_logits_only_pool,
                        plot_folder=plot_folder
                    )
                    train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx = \
                        polyadic_sampler.set_data_parameters_and_generate(polyadic_sampler_cfg)
                else:
                    # Specify own GP model for dataset generation
                    dataset_mean_module = gpytorch.means.ConstantMean()
                    dataset_base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                    dataset_likelihood = gpytorch.likelihoods.GaussianLikelihood()

                    dataset_mean_module.constant = torch.tensor([dataset_mean_constant])
                    dataset_base_kernel.base_kernel.lengthscale = dataset_length_scale
                    dataset_base_kernel.outputscale = dataset_output_scale
                    dataset_likelihood.noise_covar.noise = dataset_noise_std ** 2

                    points_initial = 10
                    dumi_train_x = torch.randn(points_initial, input_dim)
                    dumi_train_y = torch.zeros(points_initial)
                    dataset_model = polyadic_sampler.CustomizableGPModel(
                        dumi_train_x, dumi_train_y, dataset_mean_module,
                        dataset_base_kernel, dataset_likelihood
                    )

                    polyadic_sampler_cfg = polyadic_sampler.PolyadicSamplerConfig(
                        no_train_points=no_train_points, no_test_points=no_test_points,
                        no_pool_points=no_pool_points, model_name=dataset_model_name,
                        no_anchor_points=no_anchor_points, input_dim=input_dim,
                        stdev_scale=stdev_scale, stdev_pool_scale=stdev_pool_scale,
                        scaling_factor=scaling_factor, scale_by_input_dim=scale_by_input_dim,
                        model=dataset_model, stdev_blr_w=stdev_blr_w, stdev_blr_noise=stdev_blr_noise,
                        logits=logits, if_logits=if_logits, if_logits_only_pool=if_logits_only_pool,
                        plot_folder=plot_folder
                    )
                    train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx = \
                        polyadic_sampler.set_data_parameters_and_generate(polyadic_sampler_cfg)

            train_x = train_x.to(device)
            train_y = train_y.to(device)

    else:
        raise ValueError("Must provide either --config_file_path or --csv_directory")

    print(f"Training data shape: train_x={train_x.shape}, train_y={train_y.shape}")

    # Set a separate seed for predictor training to ensure reproducibility
    torch.manual_seed(args.seed_predictor)
    np.random.seed(args.seed_predictor)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed_predictor)
        torch.cuda.manual_seed_all(args.seed_predictor)

    # Train predictor
    print("\nTraining predictor network...")
    hidden_dims = [64, 32]
    output_size = 1
    dropout = 0.1

    model_predictor = train_predictor(
        train_x=train_x,
        train_y=train_y,
        device=device,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_size=output_size,
        dropout=dropout,
        lr=1e-3,
        weight_decay=1e-4,
        n_epochs=100,
        batch_size=min(32, train_x.size(0)),
        val_split=0.2,
        early_stopping_patience=10,
        verbose=True
    )

    # Save predictor
    save_predictor(
        model_predictor,
        args.output_path,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_size=output_size,
        dropout=dropout
    )

    print(f"\nPredictor trained and saved to: {args.output_path}")
    print("You can now use this predictor in all pipeline scripts by specifying --predictor_path")

    wandb.finish()


if __name__ == "__main__":
    main()
