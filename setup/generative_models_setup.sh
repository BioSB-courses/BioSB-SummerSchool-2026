### Steps to set up environments in SURF Research Cloud

# Install miniconda3 in /opt for all users
# https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30668784/Manual+use+of+miniconda
# https://askubuntu.com/questions/1508169/how-can-i-install-miniconda-to-be-accessible-to-all-users
cd /opt
# Download the installation script
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run the installation script: "-b" skips all confirmations with "yes"
sudo bash Miniconda3-latest-Linux-x86_64.sh -b -u -p miniconda3
# Accept the terms of service (tos) for all used channels.
sudo /opt/miniconda3/bin/conda tos accept
# Update conda
sudo /opt/miniconda3/bin/conda update conda -y

# Unpack environment into directory `my_env`
sudo mkdir /data/generative_models
cd /data/generative_models
sudo mkdir -p esmfold
sudo cp /home/pmoerland/researchdrive/Research_Drive_Perry_Moerland\ \(Projectfolder\)/BioSB_summer_school/Generative_models/conda-envs-amelia/esmfold.tar.gz .	
sudo tar -xzf esmfold.tar.gz -C esmfold

source esmfold/bin/activate
sudo esmfold/bin/python esmfold/bin/conda-unpack
source esmfold/bin/deactivate

sudo mkdir -p protein-design
sudo cp /home/pmoerland/researchdrive/Research_Drive_Perry_Moerland\ \(Projectfolder\)/BioSB_summer_school/Generative_models/conda-envs-amelia/protein-design.tar.gz .	
sudo tar -xzf protein-design.tar.gz -C protein-design

source protein-design/bin/activate
sudo protein-design/bin/python protein-design/bin/conda-unpack
source protein-design/bin/deactivate

sudo rm esmfold.tar.gz protein-design.tar.gz

# Then, move environments to /opt/miniconda3/envs
sudo mv protein-design esmfold /opt/miniconda3/envs/

# Git clone RFdiffusion, ProteinMPNN, ESMFold in /data/generative_models
cd /data/generative_models
if [[ ! -d "RFdiffusion" ]]; then 
	sudo git clone https://github.com/RosettaCommons/RFdiffusion.git
fi
if [[ ! -d "RFdiffusion/models" ]]; then
	sudo chmod u+x ./RFdiffusion/scripts/download_models.sh;
	sudo ./RFdiffusion/scripts/download_models.sh RFdiffusion/models
fi
if [[ ! -d "ProteinMPNN" ]]; then
	sudo git clone https://github.com/dauparas/ProteinMPNN.git
fi
if [[ ! -d "ESMFold" ]]; then
	sudo git clone https://github.com/sara-nl/ESMFold_Snellius ESMFold
fi

# Initialize conda
/opt/miniconda3/bin/conda init
# Close the current terminal and start a new one.
# Try the version parameter
conda --version

# Pip install RFdiffusion in protein-design environment
conda activate protein-design

# Run conda-unpack again to fix paths
sudo /opt/miniconda3/envs/protein-design/bin/python \
  /opt/miniconda3/envs/protein-design/bin/conda-unpack
sudo /opt/miniconda3/envs/esmfold/bin/python \
  /opt/miniconda3/envs/esmfold/bin/conda-unpack

sudo /opt/miniconda3/envs/protein-design/bin/python -m pip install --no-deps -e \
  /data/generative_models/RFdiffusion/env/SE3Transformer
sudo /opt/miniconda3/envs/protein-design/bin/python -m pip install --no-deps -e \
  /data/generative_models/RFdiffusion/

# Problem with DGL and cuda (while running RFdiffusion):
# `OSError: libcusparse.so.11: cannot open shared object file: No such file or directory`
# Fix install cuda
sudo /opt/miniconda3/bin/conda install -y -n protein-design -c conda-forge cudatoolkit=11.6 cudnn=8.4
sudo /opt/miniconda3/bin/conda env config vars set \
  -n protein-design LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Provide permissions for the RFdiffusion `schedules` folder
sudo mkdir -p /data/generative_models/RFdiffusion/schedules
sudo chmod 1777 /data/generative_models/RFdiffusion/schedules

# Download ESMFold weights only once
sudo mkdir -p /data/generative_models/ESMFold/hf-cache
sudo HF_HOME=/data/generative_models/ESMFold/hf-cache \
  /opt/miniconda3/envs/esmfold/bin/python -c \
  "from huggingface_hub import snapshot_download; \
   snapshot_download('facebook/esmfold_v1')"

# Reactivate env
conda deactivate
conda activate protein-design

# Change kernel.json to connect Jupyter to local environment by default
sudo cp /home/pmoerland/researchdrive/Research_Drive_Perry_Moerland\ \(Projectfolder\)/BioSB_summer_school/Generative_models/conda-envs-amelia/kernel.json /usr/local/share/jupyter/kernels/src-default/kernel.json
