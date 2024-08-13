import numpy as np
import matplotlib.pyplot as plt

from torchrf.utils.dataset import create_cir_dataset, load_cir_dataset


def main():
    kwargs = dict(save_filename='torchrf_cir_anechoic.h5',
                  target_num_cirs=500,
                  batch_size_cir=100,
                  scene_name="anechoic")
    create_cir_dataset(**kwargs)

    cir = load_cir_dataset(kwargs["save_filename"])

    n_dims = 3
    n_plots = n_dims**2
    fig, ax = plt.subplots(n_dims, n_dims, figsize=(5*n_dims, 5*n_dims))
    ax = ax.flatten()
    plot_indexes = np.random.choice(np.arange(len(cir)), replace=False, size=(n_plots,))
    for i, p_idx in enumerate(plot_indexes):
        ax[i].plot(np.abs(cir[p_idx]))
    plt.show()

    print("done")


if __name__ == "__main__":
    main()
