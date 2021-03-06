#!/usr/bin/env bash

VENVNAME=Computer_Vision03

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
pip install ipython
pip install jupyter
pip install tensorflow
pip install matplotlib
pip install opencv-python
pip install pydot
pip install tqdm
python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt
deactivate
echo "build $VENVNAME"
