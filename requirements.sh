#!/bin/bash
# requirements as script

# Creating python venv
sudo ln -s /bin/python3 /bin/python
sudo apt install python3-venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python -m venv venv
source venv/bin/activate

# Installing dependencies

pip install opencv-python==4.9.0.80
pip install ultralytics==8.1.43
pip install plotly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jupyter
pip install ipython
pip install ipykernel
pip install scipy
pip install notebook
ipython kernel install --user --name BallMill3D
