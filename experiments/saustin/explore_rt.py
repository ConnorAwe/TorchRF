import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle as pkl
import matplotlib

import torchrf
from torchrf.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial, Camera
from torchrf.rt.solver_paths import PathsTmpData
from torchrf.utils.tensors import expand_to_rank
from torchrf.utils.channel import cir_to_time_channel, time_lag_discrete_time_channel
from torchrf.utils.datalogger import DataLogger

def main():
    DataLogger().set_mode('print')
    DataLogger.load("logs/index.json")

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    scene = load_scene("../../src/torchrf/rt/scenes/anechoic_4/untitled.xml")
    # scene = load_scene(torchrf.rt.scene.munich)
    nop = 1
    resolution = [480, 320]

    # my_cam = Camera("my_cam", position=[-250, 250, 150], look_at=[-15, 30, 28])
    # scene.add(my_cam)
    # scene.render("my_cam", resolution=resolution, num_samples=512)
    # plt.show()

    n = 4
    i = 0
    my_cam = Camera(f"my_cam", position=[2 * 1, 4 * 2, -2 * 2], look_at=[0, 0, 0])
    scene.add(my_cam)
    scene.render(f"my_cam", resolution=resolution, num_samples=512);  # Increase num_samples to increase image quality
    scene.preview()

    # scene.render_to_file(camera="my_cam", # Also try camera="preview"
    #                      num_samples=512,
    #                      filename="scene_ball.png",
    #                      resolution=[650 ,500])
    # fig = scene.render(camera="my_cam", # Also try camera="preview"
    #                      show_paths=True,
    #                      show_devices=True,
    #                      resolution=[650 ,500])
    # plt.show()

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901",
                                 polarization="V")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901",
                                 polarization="V")

    # Create transmitter
    tx = Transmitter(name="tx",
                     position=[0.4, 0.8, 0.6])

    # Add transmitter instance to scene
    scene.add(tx)

    # Create a receiver
    n = 1
    rx = Receiver(name="rx",
                  position=[n + 0.1,n +  0.2, 0.3],
                  orientation=[0, 0, 0])

    # Add receiver instance to scene
    scene.add(rx)
    tx.look_at(rx)  # Transmitter points towards receiver
    scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterials

    scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)

    # Compute propagation paths
    paths = scene.compute_paths(max_depth=5,
                                diffraction=True,
                                scattering=True,
                                edge_diffraction=True,
                                num_samples=10e6,
                                scat_keep_prob=0.5)  # Number of rays shot into directions defined

    #
    # scene.render_to_file(camera="my_cam", # Also try camera="preview"
    #                      num_samples=512,
    #                      filename="scene_ball.png",
    #                      resolution=[650 ,500])
    # fig = scene.render(camera="my_cam", # Also try camera="preview"
    #                      show_paths=True,
    #                      show_devices=True,
    #                      resolution=[650 ,500])
    # plt.show()
    #
    # with DataLogger.push('script') as logger:
    #
    #     # Compute propagation paths
    #     paths = scene.compute_paths(max_depth=5,
    #                                 diffraction=True,
    #                                 scattering=True,
    #                                 num_samples=2e3,
    #                                 scat_keep_prob=0.5)  # Number of rays shot into directions defined
    #     logger.write(paths, 'paths')
    #
    # scene.render_to_file(camera="my_cam", # Also try camera="preview"
    #                      paths=paths,
    #                      show_paths=True,
    #                      show_devices=True,
    #                      filename="scene.png",
    #                      resolution=[650 ,500])
    # scene.render(camera="my_cam", # Also try camera="preview"
    #                      paths=paths,
    #                      show_paths=True,
    #                      show_devices=True,
    #                      resolution=[650 ,500])
    # plt.show()


    bandwidth = 15e6
    a, tau = paths.cir()
    l_min, l_max = time_lag_discrete_time_channel(bandwidth)
    h = cir_to_time_channel(bandwidth, a ,tau, l_min, l_max, normalize=True)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(abs(h.flatten().detach().numpy()), label='TorchRF CIR')
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Index")
    ax.legend()
    plt.tight_layout()
    plt.show()
    nop = 1
    nop = 1

    save_fname = "sionna_cir.pkl"
    with open(save_fname, 'rb') as f:
        sionna_h = pkl.load(f)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(abs(h.flatten().detach().numpy()), label='TorchRF CIR')
    ax.plot(abs(sionna_h), '--', label='Sionna CIR')
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Index")
    ax.legend()
    plt.tight_layout()
    plt.show()
    nop = 1

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    main()