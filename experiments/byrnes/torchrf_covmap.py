import torch
import torchrf
from torchrf.rt import load_scene, PlanarArray, Transmitter, Receiver
from torchrf.utils.datalogger import DataLogger
logger = DataLogger().set_mode('print')
DataLogger.load("logs/index.json")  # TODO: why doesn't it auto load?
scene = load_scene(torchrf.rt.scene.munich)

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                             num_cols=2,
                             vertical_spacing=0.7,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="VH")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")
# Add a transmitters
tx = Transmitter(name="tx",
                 position=[8.5,21,30],
                 orientation=[0,0,0])
scene.add(tx)

for i, target in enumerate(([40, 80, 1.5], [40, 80, 0.0], [40, 0.0, 1.5])):
    with logger.push(f"loop_{i}"):
        logger.compare(target, f"target")
        tx.look_at(torch.tensor(target))
        cm = scene.coverage_map(cm_cell_size=[1.,1.], num_samples=int(10e3))  # 10e6
        logger.compare(cm.as_tensor(), f"map", 1e-11)
        logger.compare(cm.as_tensor().sum().item(), f"sum")
