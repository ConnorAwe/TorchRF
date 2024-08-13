"""
This notebook demonstrates use of the TorchRF DataLogger.  The DataLogger allows a user to record intermediate results from a computation (for example, in Sionna) and compare those to intermediate results in a separate computation (for example, in TorchRF).  The calls to save out the data need to be made where the data is available, which might be inside of a function which could be called multiple times.  In order to align saved output, a sequence of tags similar to the call stack is used to name the saved variables.

In the basic use case, two separate runners are used for the two separate runs.  The first runner should save outputs and the second runner should load the saved outputs and compare them to newly generated ones.  We demonstrate this in two notebooks.  The current notebook, `LogDemoA`, demonstrates writing.  The second notebook, `LogDemoB`, demonstrates loading and comparing.
"""

"""
Import the DataLogger and set the mode to 'write'.
"""
from torchrf.utils.datalogger import DataLogger
DataLogger().set_mode('write')

"""
Proceed with runner for the reference computation
"""
import numpy as np
import sionna
from sionna.rt.scene import load_scene
from sionna.rt import PlanarArray, Transmitter, Receiver, Camera

scene = load_scene(sionna.rt.scene.munich)
resolution = [480, 320]
my_cam = Camera("my_cam", position=[-250, 250, 150], look_at=[-15, 30, 28])
scene.add(my_cam)
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                             pattern="tr38901",polarization="V")
tx = Transmitter(name="tx", position=[8.5, 21, 27])
scene.add(tx)

"""
The DataLogger is a singleton class.  The instance can be obtained as if by a typical constructor, but the same 
instance is returned every time.
"""
logger = DataLogger()
logger2 = DataLogger()
assert logger is logger2

"""
The simplest way to save out a variable is just to use the write command, which gets a name used for alignment.
"""
logger.write(tx, 'transmitter1')
""" 
This write command pickled the transmitter.  Tensorflow tensors and PyTorch tensors are converted to numpy.ndarrays before pickling.  The logger recurs through lists, dicts, and complex classes such as Transmmitter.  Because many of the classes we want to save point to Scene objects, which are large, the DataLogger ignores Scene objects.  There is nothing to stop the recursion from traversing down an infinite loop if two objects point to each other, but this does not come up when Scenes are ignored.

FUTURE: track objects to avoid looping.
FUTURE: make it easy for user to specify additional classes to ignore.
"""

"""
When the user believes that two tensors are identical up to reordering of vectors and that vectors are unique (as happens with the outputs of `tf.raw.uniqueV2` and `torch.unique`, the user can specify a "unique" keyword when comparing.  We will demonstrate this on the tensor below:
"""
a = np.array([[7, 5, 3],
              [7, 5, 3],
              [7, 5, 3]])
logger.write(a, '753array')

""" 
Write calls inside of functions might be called in different contexts, and we will need to track those contexts the same way one tracks the call stack, so that we compare the correct outputs.  The following example throws an error because it tries to save 3 different values under the same tag: 
"""
def foo(x):
    y = x * x
    logger.write(y, 'intermediate foo')
    z = y/2
    return z

z1 = foo(7)
error = False
try:
    z2 = foo(5)
except RuntimeError:
    error = True
assert error

"""
We accommodate this by stacking tags with push() and pop()
"""
logger.push('foo 1')
z1 = foo(7)
logger.pop()

logger.push('foo 2')
z2 = foo(5)
logger.pop()

logger.push('foo 3')
z3 = foo(3)
logger.pop()

"""

This is made slightly more convenient by using `with` blocks.  The end of `with` block automatically pops the logger and automatically records the index to disk if the stack is empty.  The example below is nested, which will create a different tag structure from the tags above.

"""
with DataLogger().push('foo A') as logger:  # Can't use 'foo 1' because it was used above.
    z1 = foo(7)
    logger.write(z1, 'first returned z')
    with logger.push('foo 2') as logger:  # This tag is now 'foo A:foo 2'
        z2 = foo(5)
        with logger.push('foo 3') as logger:  # This tag is now 'foo A:foo 2:foo 3'
            z3 = foo(3)

"""
Let's try some data that we know will fail to be recreated in the second run:
"""
with DataLogger().push('rand') as logger:
    a = np.random.rand(3, 5)
    logger.write(a, 'A')
    z = foo(a)
    logger.write(z, 'foo A')

"""

When the with block exits, if the stack is empty then it automatically saves the index.  So ideally you wrap all of your writes in an outermost with block in the runner, as above.  If you have a `write()` that was not in a `with`, then you'll have to call `save()` to save out the index:

"""
logger.save()
"""

In this case we did not need to call save this final time, but it only has the effect of rewriting the same information to the index file again, so it does no harm.

"""
