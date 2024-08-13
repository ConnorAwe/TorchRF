import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch.nn as nn
import torch
import numpy as np
import h5py
import tqdm

import torchrf
from torchrf.rt import load_scene, Transmitter, Receiver, PlanarArray
from torchrf.utils.channel import cir_to_time_channel, time_lag_discrete_time_channel


def load_sionna_arrays(num_rows=1,
                       num_cols=1,
                       vertical_spacing=0.5,
                       horizontal_spacing=0.5,
                       pattern="tr38901",
                       polarization="V",
                       **kwargs):
    """
    Loads an array configuration for Transmitter or Receiver in a Sionna scene
    """
    array = PlanarArray(num_rows=num_rows,
                        num_cols=num_cols,
                        vertical_spacing=vertical_spacing,
                        horizontal_spacing=horizontal_spacing,
                        pattern=pattern,
                        polarization=polarization)
    return array


def load_sionna_scene(scene_name="munich", **kwargs):
    if scene_name == 'munich':
        scene = load_scene(torchrf.rt.scene.munich)  # Try also sionna.rt.scene.etoile
    elif scene_name == 'anechoic':
        scene = load_scene("/home/saustin/saledin/torchrf/src/torchrf/rt/scenes/anechoic_2/single_ball_anechoic_chamber.xml")
    else:
        raise ValueError(f"{scene_name} is an invalid scene_name")

    scene.tx_array = load_sionna_arrays()
    scene.rx_array = load_sionna_arrays()

    return scene


def load_cir_dataset(filename, return_locations=False, dtype=torch.complex64, device='cpu'):
    """
    Loads previously generated CIR datasets and returns them as tensors
    """
    with h5py.File(filename, 'r') as hf:
        cir = np.vstack([x for x in hf['cir']])

    cir = torch.tensor(cir, dtype=dtype, device=device)

    return cir


def create_cir_dataset(target_num_cirs=5000,
                       batch_size_cir=1000,
                       scene_name='munich',
                       save_filename='cir.h5',
                       diffraction=False,
                       scattering=False,
                       min_gain_db=-130,
                       max_gain_db=0,
                       min_dist=5,
                       max_dist=400,
                       max_depth=5,
                       num_samples=1e6,
                       center_frequency=2.14e9,
                       bandwidth=18e6,
                       dtype=np.complex128,
                       device='cpu',
                       **kwargs
                       ):
    """
    Creates h5 file of CIRs derived from the Sionna lib
    """
    if scene_name == "anechoic":
        tx_position = (0.4, 0.8, 0.6)
    else:
        tx_position = (8.5, 21, 27)

    scene = load_sionna_scene(scene_name, **kwargs)
    scene.frequency = center_frequency
    scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)

    # Create transmitter
    tx = Transmitter(name='tx', position=list(tx_position))
    scene.add(tx)

    # Update coverage_map
    cm = scene.coverage_map(max_depth=max_depth,
                            diffraction=diffraction,
                            scattering=scattering,
                            cm_cell_size=(1., 1.),
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(num_samples))

    if scene_name == "anechoic":
        # Chamber Is aligned at (0, 0, 0)
        # Chamber has dimensions (1.5m, 1.2m, 1.2m)
        ue_pos = np.stack((np.random.uniform(low=0.1, high=1.4, size=(batch_size_cir,)),
                           np.random.uniform(low=0.1, high=1.1, size=(batch_size_cir,)),
                           np.random.uniform(low=0.1, high=1.1, size=(batch_size_cir,))), axis=-1)
    else:
        # sample batch_size random user positions from coverage map
        ue_pos = cm.sample_positions(batch_size=batch_size_cir,
                                     min_gain_db=min_gain_db,
                                     max_gain_db=max_gain_db,
                                     min_dist=min_dist,
                                     max_dist=max_dist)

    # Create batch_size receivers
    for i in range(batch_size_cir):
        rx = Receiver(name=f"rx-{i}",
                      position=ue_pos[i],  # Random position sampled from coverage map
                      )
        scene.add(rx)

    # Get lag times
    l_min, l_max = time_lag_discrete_time_channel(bandwidth)

    # Placeholder to gather channel impulse reponses
    h_time = []

    # Each simulation returns batch_size_cir results
    num_runs = int(np.ceil(target_num_cirs / batch_size_cir))
    for idx in tqdm.tqdm(range(num_runs)):
        print(f"Progress: {idx + 1}/{num_runs}", end="\r")

        if scene_name == "anechoic":
            # Chamber Is aligned at (0, 0, 0)
            # Chamber has dimensions (1.5m, 1.2m, 1.2m)
            ue_pos = np.stack((np.random.uniform(low=0.1, high=1.4, size=(batch_size_cir,)),
                               np.random.uniform(low=0.1, high=1.1, size=(batch_size_cir,)),
                               np.random.uniform(low=0.1, high=1.1, size=(batch_size_cir,))), axis=-1)
        else:
            # sample batch_size random user positions from coverage map
            ue_pos = cm.sample_positions(batch_size=batch_size_cir,
                                         min_gain_db=min_gain_db,
                                         max_gain_db=max_gain_db,
                                         min_dist=min_dist,
                                         max_dist=max_dist)

        # Update all receiver positions
        for j in range(batch_size_cir):
            scene.receivers[f"rx-{j}"].position = ue_pos[idx]

        # Simulate CIR
        paths = scene.compute_paths(
            max_depth=max_depth,
            diffraction=diffraction,
            scattering=scattering,
            num_samples=int(num_samples))  # shared between all tx in a scene

        # We fix here the maximum number of paths to 75 which ensures
        # that we can simply concatenate different channel impulse reponses
        a_, tau_ = paths.cir()
        del paths  # Free memory

        h_time.append(cir_to_time_channel(bandwidth, a_, tau_, l_min, l_max, normalize=True).detach().cpu().numpy()[0, :, 0, 0, 0, 0])

    del cm  # Free memory

    h_time = np.vstack([h[np.sum(np.abs(h) ** 2, axis=-1) > 0] for h in h_time])
    with h5py.File(save_filename, 'w') as hf:
        hf.create_dataset(f'cir', data=h_time)
        hf.close()

