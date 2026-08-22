# Practical 2 - Generative models

For this session we will use workspaces (aka as virtual machines) on SURF Research Cloud. There are four workspaces (Ubuntu 22.04, A10 - 2 GPU) available and each of you has been assigned to one of the workspaces. Open [generative_models_workspace_assignments.md](generative_models_workspace_assignments.md) and then click the link behind your name to log in and go to your designated workspace.

## Initial setup

Open a terminal in the machine. Initialize conda once to add it to your .bashrc

```
/opt/miniconda3/bin/conda init
```

Close terminal and open a new one. Check conda with `conda --version`

Clone the GitHub repository in your home directory and copy the files for *Practical 2 - Generative models*

```
git clone https://github.com/BioSB-courses/BioSB-SummerSchool-2026.git
cp -r BioSB-SummerSchool-2026/Practical-2-Generative-models/* .
```

This copies two notebooks `protein_design_motif_scaffolding.ipynb`, `protein_design_unconditional.ipynb` and a `scripts` directory.


## Practical session steps

We will use RFdiffusion to generate structures, evaluate them, and then try to find sequences that match this structure. Start from `protein_design_unconditional.ipynb`
