from typing import NamedTuple
import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from uqct.eval.dense import DiffusionRecon, schedule_exponential, ObservationDataset, fbp_recon, psnr, BootstrapRecon, guidance_loss, CondDiffusionRecon, guidance_loss_diverse, load_diffusion_unet, DDPMScheduler, fbp, sinogram_from_counts
from uqct.datasets.utils import get_dataset
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
import numpy as np
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from tqdm.auto import tqdm
import uqct


def read_dataset(*paths, dataset=None):
    # recursively read all netcdf files in path
    nc_files = []
    for path in paths:
        nc_files.extend(sorted(glob.glob(f"{path}/**/*.nc", recursive=True)))

    datasets = {}
    attrs = []
    for f in nc_files:
        # print(f)
        ds = xr.open_dataset(f)
        # print(ds.attrs)

        if dataset is not None:
            if ds.attrs["dataset"] != dataset:
                continue
        attrs.append(ds.attrs)
        experiment_id = ds.attrs['experiment_id']
        if not 'beta' in ds:
            ds['beta'] = ds['seq_nll'].cumsum(dim='step')
        # else:
        #     print((ds['beta'] - ds['seq_nll'].cumsum(dim='step')).max())
        if 'seq_nll_mix' in ds:
            if not 'beta_mix' in ds:
                ds['beta_mix'] = ds['seq_nll_mix'].cumsum(dim='step')
            # else:
            #     print((ds['beta_mix'] - ds['seq_nll_mix'].cumsum(dim='step')).max())
        datasets[experiment_id] = ds
    df = pd.DataFrame(attrs)
    df = df.set_index('experiment_id')
    return datasets, df

def find_experiment(attrs, datasets, dataset, model, samples=False, aggregate_seeds=False, match_attrs=None):
    if 'samples' in attrs.columns:
        matching_experiments = attrs.loc[(attrs['model'] == model) & (attrs['dataset'] == dataset) & (attrs['samples'] == samples)]
    else:
        matching_experiments = attrs.loc[(attrs['model'] == model) & (attrs['dataset'] == dataset)]

    if match_attrs is not None:
        for key, value in match_attrs.items():
            matching_experiments = matching_experiments.loc[matching_experiments[key] == str(value)]
    if len(matching_experiments) == 0:
        raise ValueError(f"No experiment found for model {model} and dataset {dataset}.")
    elif aggregate_seeds:
        if len(matching_experiments) == len(matching_experiments['seeds'].unique()):
            # print(f"INFO Aggregating {len(matching_experiments)} experiments for model {model} and dataset {dataset}.")
            # print(f"Aggregation {len(matching_experiments)} experiments for model {model} and dataset {dataset}.")

            all_ds = []
            for experiment_id in matching_experiments.index:
                ds = datasets[experiment_id].sel(model=model)
                all_ds.append(ds)
            ds = xr.concat(all_ds, dim='seed', data_vars='all')
            # sort seed dimension
            ds = ds.sortby('seed')
            return ds, matching_experiments.index.values
        else:
            raise ValueError(f"WARNING Number of matching experiments {len(matching_experiments)} does not match number of unique seeds {len(matching_experiments['seeds'].unique())}. Not aggregating.")
    elif len(matching_experiments) > 1:
        raise ValueError(f"WARNING Multiple experiments found for model {model} and dataset {dataset}.")

    experiment_id = matching_experiments.index[0]
    ds = datasets[experiment_id].sel(model=model)
    return ds, experiment_id

def print_stats(attrs):
    for initial_intensity, total_intensity in [
        (1e4, 1e9),
        (1e4, 1e5),
        (1e5, 1e6),
        (1e6, 1e7),
        (1e7, 1e8),
        (1e8, 1e9)
    ]:
        mask = (attrs['initial_intensity'] == str(initial_intensity)) & (attrs['total_intensity'] == str(total_intensity))
        if mask.sum() > 0:
            print(f"{initial_intensity:.0e} to {total_intensity:.0e}: {mask.sum()} experiments")
            # Count the number of rows for each "dataset" and "model" in attrs
            count_df = attrs[mask].groupby(['model', 'dataset']).size().unstack(fill_value=0)
            print(count_df)

# read final experiment runs
datasets, attrs = read_dataset("/mydata/chip/johannes/uq-xray-ct/results-final/2026-01-24")
print_stats(attrs)

MATCH_ATTRS = {
    'cond_diffusion' : {'guidance_num_gradient_steps' : 10},
    'diverse_cond_diffusion' : { 'guidance_lr' : 1e-3, 'guidance_num_gradient_steps' : 10 },
    'diffusion' : { 'guidance_lr' : 1e-1, 'guidance_num_gradient_steps' : 10 },
}

class Args(NamedTuple):
    model: str = "fbp"  # "unet", "diffusion", "cond_diffusion"
    dataset: str = "lung"
    num_images: int = 10
    batch_size: int = 2
    seeds: list[int] = [0]
    schedule: str = "exponential"  # "uniform" or "exponential"
    total_intensity: float = 1e9
    num_steps: int = 10
    init_fraction: float = None
    initial_intensity: float = 1e7
    base: float = 1.1
    num_samples: int = 5
    diffusion_num_inference_steps: int = 100
    guidance_end: int = 10
    guidance_num_gradient_steps: int = 10
    guidance_lr: float = 1e-3
    guidance_lr_decay: bool = False

def compute_samples(dataset='lamino'):
    model = "cond_diffusion"
    seed = 0

    ds, _ = find_experiment(attrs, datasets, dataset, model, aggregate_seeds=True, match_attrs=MATCH_ATTRS[model])

    args = Args(
        num_samples=16,
        model=model,
        seeds=[seed], 
        dataset=dataset, 
        initial_intensity=float(ds.attrs['initial_intensity']), 
        total_intensity=float(ds.attrs['total_intensity']), 
        num_steps=int(ds.attrs['num_steps']),
        diffusion_num_inference_steps=int(ds.attrs['diffusion_num_inference_steps']),
        guidance_end=int(ds.attrs['guidance_end']),
        guidance_num_gradient_steps=int(ds.attrs['guidance_num_gradient_steps']),
        guidance_lr=float(ds.attrs['guidance_lr']),
        guidance_lr_decay=bool(ds.attrs['guidance_lr_decay']),
        )


    _ , test_set = get_dataset(args.dataset, True)
    test_set = torch.utils.data.Subset(test_set, list(range(10, 110)))

    num_angles = 200
    angles = torch.from_numpy(np.linspace(0, 180, num_angles, endpoint=False)).float().to(device)

    schedule = schedule_exponential(
        1e9, 30, initial_intensity=args.initial_intensity, device=device
    )
    total_intensities = schedule.clone()
    # print(total_intensities.cumsum(dim=0))
    schedule = schedule.reshape(-1, 1, 1, 1).expand(-1, 1, num_angles, 1) / num_angles / 256
        
    obs_dataset = ObservationDataset(test_set, seeds=[0], intensities=schedule, angles=angles)
    obs_dataloader = torch.utils.data.DataLoader(
        obs_dataset, batch_size=5, shuffle=False, num_workers=0
    )


    # bootstrap_recon = BootstrapRecon(fbp_recon, num_samples=20)

    ckpt_path = Path(f"/mydata/chip/shared/checkpoints/uqct/diffusion/ddpm_conditional_128_{args.dataset}.pt")
    unet = load_diffusion_unet(ckpt_path, cond=True)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")


    cond_diffusion_recon = CondDiffusionRecon(
                unet,
                scheduler,
                num_samples=args.num_samples,
                num_inference_steps=50,
                guidance_start=1000,
                guidance_end=0,
                guidance_num_gradient_steps=10,
                guidance_lr=args.guidance_lr,
                guidance_lr_decay=False,
                seed=0,
                guidance_loss=guidance_loss
            )

    distance_diffusion_recon = CondDiffusionRecon(
                unet,
                scheduler,
                num_samples=args.num_samples,
                num_inference_steps=50,
                guidance_start=1000,
                guidance_end=0,
                guidance_num_gradient_steps=10,
                guidance_lr=args.guidance_lr,
                guidance_lr_decay=False,
                seed=0,
                guidance_loss=guidance_loss_diverse
            )

    ckpt_path = Path(f"/mydata/chip/shared/checkpoints/uqct/diffusion/ddpm_unconditional_128_{args.dataset}.pt")
    unet = load_diffusion_unet(ckpt_path, cond=False)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    diffusion_recon_10 = DiffusionRecon(
        unet,
        scheduler,
        num_samples=args.num_samples,
        num_inference_steps=args.diffusion_num_inference_steps,
        guidance_start=1000,
        guidance_end=0,
        guidance_num_gradient_steps=10,
        guidance_lr=1e-1,
        guidance_lr_decay=True,
        seed=0
    )

    diffusion_recon_5 = DiffusionRecon(
        unet,
        scheduler,
        num_samples=args.num_samples,
        num_inference_steps=args.diffusion_num_inference_steps,
        guidance_start=1000,
        guidance_end=0,
        guidance_num_gradient_steps=5,
        guidance_lr=1e-1,
        guidance_lr_decay=True,
        seed=0
    )

    beta_ds = ds


    for indices, seed, images, data in tqdm(obs_dataloader):
        images = images.to(device)
        images_lr = F.interpolate(images, size=(128, 128), mode="area")
        data = data.to(device)

        data_cumsum = data.cumsum(dim=1)
        schedule_cumsum = schedule.unsqueeze(0).cumsum(dim=1)

        with torch.no_grad():
            pred_fbp = [fbp_recon(data_cumsum[:, i], schedule_cumsum[:, i], angles) for i in range(data.shape[1])]

            pred_cond_diffusion = [cond_diffusion_recon(data_cumsum[:, i], schedule_cumsum[:, i], angles, verbose=False) for i in range(data.shape[1])]

            pred_diffusion_10 = [diffusion_recon_10(data_cumsum[:, i], schedule_cumsum[:, i], angles, verbose=False) for i in range(data.shape[1])]

            pred_diffusion_5 = [diffusion_recon_5(data_cumsum[:, i], schedule_cumsum[:, i], angles, verbose=False) for i in range(data.shape[1])]


            indices_np = indices.cpu().numpy()
            seeds_np = seed.cpu().numpy()

            # Build tuples for selection
            pairs = list(zip(indices_np, seeds_np))

            # Select using .sel with a list of tuples
            beta = beta_ds['beta_mix'].stack(sample=("index", "seed")).sel(sample=pairs).squeeze().transpose('sample', 'step')

            beta = torch.from_numpy(beta.values).to(device).float()

            data_steps = data
            schedule_steps = schedule.unsqueeze(0)

            beta_delta = beta + torch.log(1/torch.tensor(0.05))
            pred_distance = [distance_diffusion_recon(data_cumsum[:, i], schedule_cumsum[:, i], angles, beta=beta_delta[:, i], data_steps=data_steps[:, :i+1], schedule_steps=schedule_steps[:, :i+1], verbose=False) for i in range(data.shape[1])]


        pred_fbp = torch.stack(pred_fbp, dim=-4)
        pred_cond_diffusion = torch.stack(pred_cond_diffusion, dim=-4)
        pred_diffusion_10 = torch.stack(pred_diffusion_10, dim=-4)
        pred_diffusion_5 = torch.stack(pred_diffusion_5, dim=-4)
        pred_distance = torch.stack(pred_distance, dim=-4)




        break  # compute single batch only


    return {
        'fbp': pred_fbp,
        'seeds' : seed,
        'indices' : indices,
        'cond_diffusion': pred_cond_diffusion,
        'diffusion_10': pred_diffusion_10,
        'diffusion_5': pred_diffusion_5,
        'distance': pred_distance,
        'images': images,
        'images_lr': images_lr,
        'data': data,
        'schedule': schedule,
        'angles': angles,
        'beta': beta,
    }

import os
for dataset in ['lamino', 'composite', 'lung']:
    print(f"Processing dataset: {dataset}")
    predictions = compute_samples(dataset=dataset)
    output_dir = Path(f"/mydata/chip/johannes/uq-xray-ct/results-samples/{dataset}/")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")

    for key, value in predictions.items():
        print(key, value.shape)
        torch.save(value.cpu(), output_dir / f"{key}.pt")

