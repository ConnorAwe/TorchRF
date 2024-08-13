# TorchRF

TorchRF is a python library used to simulate and generated RF channel effects. It is based off of the Sionna python 
library (https://nvlabs.github.io/sionna/index.html), but is implemented with pytorch instead of Tensorflow to allow for 
easier integration into machine learning pipelines that make use of Generative adversarial techniques. 

## Getting started
Install the library
```commandline
python -m venv venv
pip install -e .
```

## Run the Tutorial
The jupyter lab tutorial is found in the tutorial file. Shows how to generate a simulated channel impulse response using 
the adapted pytorch ray tracing modules.

