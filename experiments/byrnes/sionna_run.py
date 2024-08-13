import sionna
from sionna.rt.scene import load_scene
from sionna.rt import PlanarArray, Transmitter, Receiver, Camera
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
starttime = time.time()
from torchrf.utils.datalogger import DataLogger
logger = DataLogger().set_mode('write')

scene = load_scene(sionna.rt.scene.munich)
resolution = [480, 320]
print(f"{time.time()-starttime:5.0f}s: Loaded scene")
my_cam = Camera("my_cam", position=[-250, 250, 150], look_at=[-15, 30, 28])
scene.add(my_cam)
# scene.render("my_cam", resolution=resolution, num_samples=512)

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
                 position=[8.5, 21, 27])

# Add transmitter instance to scene
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx",
              position=[45, 90, 1.5],
              orientation=[0, 0, 0])

# Add receiver instance to scene
scene.add(rx)
tx.look_at(rx)  # Transmitter points towards receiver
scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterials
print(f"{time.time()-starttime:5.0f}s: Added Tx -> Rx")

# Compute propagation paths
num_samples = 2000
with logger.push('script') as logger:
    paths = scene.compute_paths(max_depth=5,
                                num_samples=num_samples)  # Number of rays shot into directions defined
    logger.write(paths, 'paths')

print(f"{time.time()-starttime:5.0f}s: Computed paths using {num_samples} samples")

if False:
    scene.render_to_file(camera="my_cam",  # Also try camera="preview"
                         paths=paths,
                         show_paths=True,
                         show_devices=True,
                         filename="sionna_scene.png",
                         resolution=[650,500])
    print(f"{time.time()-starttime:5.0f}s: Rendered Scene")
