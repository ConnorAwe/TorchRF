#!/bin/bash

_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_DIR"

#!/bin/bash
cp -v .gitignore $1/.gitignore
cp -v requirements.txt $1/requirements.txt
cp -v setup.py $1/setup.py

cp -v src/torchrf/*.py $1/src/torchrf/
cp -v src/torchrf/rt/*.py $1/src/torchrf/rt/
cp -v src/torchrf/channel/*.py $1/src/torchrf/channel/
cp -v src/torchrf/env/*.py $1/src/torchrf/env/
cp -v src/torchrf/utils/*.py $1/src/torchrf/utils/

cp -r src/torchrf/rt/scenes/c4 $1/src/torchrf/rt/scenes/
cp -r src/torchrf/rt/scenes/c5 $1/src/torchrf/rt/scenes/
cp -r src/torchrf/rt/scenes/s_big $1/src/torchrf/rt/scenes/
cp -r src/torchrf/rt/scenes/sg_big $1/src/torchrf/rt/scenes/

cp -v tutorials/rt_example.ipynb $1/tutorials/rt_example.ipynb
cp -v tests/*.py $1/tests/