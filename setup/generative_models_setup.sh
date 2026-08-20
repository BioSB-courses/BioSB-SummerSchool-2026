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
# conda-unpack in home directory: https://conda.github.io/conda-pack/
# where `my_env` is `protein-design` and `esmfold`
	sudo /opt/miniconda3/bin/conda install conda-pack

# playground	
	sudo /opt/miniconda3/bin/conda-pack -p /opt/miniconda3/envs/esmfold/ -o esmfold.tar.gz
	sudo chown $USER:$GROUP esmfold.tar.gz
	sudo /opt/miniconda3/bin/conda-pack -p /opt/miniconda3/envs/protein-design/ -o protein-design.tar.gz --ignore-missing-files
	sudo chown $USER:$GROUP protein-design.tar.gz

	# Unpack environment into directory `my_env`
	sudo mkdir /data/generative_models
	cd /data/generative_models
	sudo mkdir -p esmfold
	sudo tar -xzf esmfold.tar.gz -C esmfold
	sudo mkdir -p protein-design
	sudo tar -xzf protein-design.tar.gz -C protein-design

# Then, move environments to /opt/miniconda3/envs
	sudo mv protein-design esmfold /opt/miniconda3/envs/

# Initialize conda
	miniconda3/bin/conda init
	# Close the current terminal and start a new one.
	# Try the version parameter
	conda --version

# Git clone RFdiffusion, ProteinMPNN, ESMFold in /data/generative_models
	cd /data/generative_models
	if [[ ! -d "RFdiffusion" ]]; then 
		sudo git clone https://github.com/RosettaCommons/RFdiffusion.git
	fi
	if [[ ! -d "RFdiffusion/models" ]]; then
		./RFdiffusion/scripts/download_models.sh RFdiffusion/models
	fi
	if [[ ! -d "ProteinMPNN" ]]; then
		sudo git clone https://github.com/dauparas/ProteinMPNN.git
	fi
	if [[ ! -d "ESMFold" ]]; then
		sudo git clone https://github.com/sara-nl/ESMFold_Snellius ESMFold
	fi

# Pip install RFdiffusion in protein-design environment
	conda activate protein-design
    sudo pip install -e "/data/generative_models/RFdiffusion/env/SE3Transformer"
   	sudo pip install -e "/data/generative_models/RFdiffusion"

# Problem with DGL and cuda (while running RFdiffusion):
# `OSError: libcusparse.so.11: cannot open shared object file: No such file or directory`
# Fix install cuda
	conda activate protein-design
    sudo /opt/miniconda3/bin/conda install -y -c conda-forge cudatoolkit=11.6 cudnn=8.4
    sudo /opt/miniconda3/bin/conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Reactivate env
	conda deactivate
	conda activate protein-design

# Change kernel.json to connect Jupyter to local environment by default
	sudo nano /usr/local/share/jupyter/kernels/src-default/kernel.json

{
 "argv": [
  "/opt/miniconda3/envs/protein-design/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python (protein-design via src-default)",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
