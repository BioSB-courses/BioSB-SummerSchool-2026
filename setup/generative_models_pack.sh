# # conda-unpack in home directory: https://conda.github.io/conda-pack/
# where `my_env` is `protein-design` and `esmfold`
sudo /opt/miniconda3/bin/conda install conda-pack	
sudo /opt/miniconda3/bin/conda-pack -p /opt/miniconda3/envs/esmfold/ -o esmfold.tar.gz
sudo chown $USER:$GROUP esmfold.tar.gz
sudo /opt/miniconda3/bin/conda-pack -p /opt/miniconda3/envs/protein-design/ -o protein-design.tar.gz --ignore-missing-files
sudo chown $USER:$GROUP protein-design.tar.gz