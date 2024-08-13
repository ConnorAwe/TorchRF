"""

This notebook demonstrates use of the TorchRF DataLogger.  See LogDemoA for an overview and examples of writing out reference values.  In this notebook we load in those saved values and compare them to values created here.

"""

"""

Import the DataLogger. The default mode is 'break', which calls a breakpoint so that you can use the debugger to inspect the differences between the current code and the reference computation.  Since we're in a Jupyter Notebook right now, we will use mode 'print' instead which just prints out disagreements.

"""
from torchrf.utils.datalogger import DataLogger
DataLogger().set_mode('print')

"""
Proceed with runner for the reference computation
"""
import numpy as np
import torchrf
from torchrf.rt.scene import load_scene
from torchrf.rt import PlanarArray, Transmitter, Camera

scene = load_scene(torchrf.rt.scene.munich)
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

"""

The simplest way to compare a variable to reference is just to call compare. You'll need to use the identical tag in order to compare to the correct variable.

"""
logger = DataLogger()
logger.compare(tx, 'transmitter1')

"""

Because the comparison was successful, we get a message that this tag "passed".  Note that this tx contains torch tensors while the reference tx contained tensorflow tensors, but both were converted to numpy ndarrays, so the comparison went through.  The comparison done on the Transmitter class was "deep", in that all fields (and fields of fields, etc) other than the Scene were compared.

"""

"""

Recall that in LogDemoA, the 753array contained a set of 3 vectors: {(3,3,3), (5,5,5), (7,7,7)} with the vectors stored in dim=1, but specfically they were stored with the 7's leftmost and the 3's rightmost.  Let's try a shuffle of those vectors:

"""
a = np.array([[5, 3, 7],
              [5, 3, 7],
              [5, 3, 7]])
logger.compare(a, '753array')

"""

As expected, that comparison fails because the tensors are not equivalent.  But if we know that we only want to know 
whether these are the same set of unique vectors, we can use the `unique` keyword.  In this case we will also need to 
specify the dim (it's the same one used as an argument to the tf or torch `unique` functions).


"""

logger.compare(a, '753array', unique=True, dim=1)

"""
This only works if the vectors really were unique, so the following fails, for example:
"""
logger.compare(np.array([[5, 7, 3, 7],
                         [5, 7, 3, 7],
                         [5, 7, 3, 7]]), '753array', unique=True, dim=1)


def foo(x):
    y = x * x
    logger.compare(y, 'intermediate foo')
    z = y / 2
    return z

"""
We build up tag stacks for reading and comparing in the same way as when writing.  The data will only be compared when the entire stack matches.

"""
with DataLogger().push('foo A') as logger:
    z1 = foo(7)
    logger.compare(z1, 'first returned z')
    with logger.push('foo 2') as logger:
        z2 = foo(5)
        with logger.push('foo 3') as logger:
            z3 = foo(3)

"""
Recall that LogDemoA wrote a random matrix.  We will obviously fail on that:
"""
with DataLogger().push('rand') as logger:
    a = np.random.rand(3, 5)
    logger.compare(a, 'A')

"""

But we might want to make sure that our function `foo` operates on that random matrix the same way that the original `foo`.  We can do that by explicitly loading in the saved data:

"""
with DataLogger().push('rand') as logger:
    a = logger.get('A')
    z = foo(a)
    logger.compare(z, 'foo A')

