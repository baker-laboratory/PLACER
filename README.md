# PLACER
#### <ins>P</ins>rotein-<ins>L</ins>igand <ins>A</ins>tomistic <ins>C</ins>onformational <ins>E</ins>nsemble <ins>R</ins>esolver

![image](https://github.com/user-attachments/assets/78d7fc0e-ab45-4879-abf5-1145a948a7fb)


(formerly known as ChemNet)

PLACER is a graph neural network that operates entirely at the atomic level; the nodes of the graph are the atoms in the system. PLACER was trained to recapitulate the correct atom positions from partially corrupted input structures from the Cambridge Structural Database and the Protein Data Bank. PLACER accurately generates structures of diverse organic small molecules given knowledge of their atom composition and bonding, and given a description of the larger protein context, can accurately build up structures of small molecules and protein side chains; used in this way PLACER has competitive performance on protein-small molecule docking given approximate knowledge of the binding site. PLACER is a rapid and stochastic denoising network, which enables generation of ensembles of solutions to model conformational heterogeneity.

Reference: https://www.biorxiv.org/content/10.1101/2024.09.25.614868v1


## Table of contents

- [Installation](#set-up)
- [Usage](#usage)
- [Examples](#examples)
- [Python API usage](#api_usage)
- [Questions & Troubleshooting](#faq)
- [Disclaimer](#disclaimer)


<a id="set-up"></a>
## Installation

Clone the repository
`git clone https://github.com/baker-laboratory/PLACER.git`

The repository contains the model weights and the code is ready to run after an appropriate Python environment is set up.

### Requirements
  [cuda-toolkit](https://anaconda.org/nvidia/cuda-toolkit) >= 12.1<br>
  [pytorch](https://github.com/pytorch/pytorch) = 2.3.*, build=*cuda12*<br>
  [dgl](https://github.com/dmlc/dgl) = 2.4.0<br>
  [opt_einsum](https://github.com/dgasmith/opt_einsum) = 3.4.0<br>
  [openbabel](https://github.com/openbabel/openbabel) = 3.1.1<br>
  [networkx](https://github.com/networkx/networkx) >= 3.2<br>
  [numpy](https://github.com/numpy/numpy) >= 1.2.6<br>
  [pandas](https://github.com/pandas-dev/pandas) = 2.2.3<br>
  [e3nn](https://github.com/e3nn/e3nn) = 0.5.4

  And for convenience: `matplotlib`, `ipython`, `jupyterlab`

### Conda environment

A minimal conda environment for running PLACER:<br>
Create a new environment from `envs/placer_env.yml`:
```
conda env create -f envs/placer_env.yml

conda activate placer_env
```

### For Mac users

Clone the forked repository for Apple Silicon:
`git clone https://github.com/fnachon/PLACER.git`

A specific conda environment is necessary for running PLACER on the gpu of Apple Silicon using mps. It is based on [SE3Transformer adapted for mps by YaoYingYing](https://github.com/YaoYinYing/SE3Transformer)<br>
Create a new environment from `envs/placer_env_mac.yml`:
```
conda env create -f envs/placer_env_mac.yml

conda activate placer
```
<a id="usage"></a>
## Usage

PLACER is available as a commandline script, and as a Python module. For Python API usage, see [below](#api_usage).

To run PLACER analysis from the commandline, use:
`python run_PLACER.py ...`

Available arguments:
```
  -h, --help            show this help message and exit
  -i IDIR, --idir IDIR  input folder with PDB/mmCIF files (default: None)
  -f IFILE, --ifile IFILE
                        file with a list of input PDB/mmCIF files or a single PDB/mmCIF file. Only mmCIF files from RCSB are correctly parsed. (default: None)
  -o ODIR, --odir ODIR  output folder to save models and CSV files. Default is current run directory. (default: ./)
  -n NSAMPLES, --nsamples NSAMPLES
                        number of samples to generate. 50-100 is a good number in most cases. (default: 10)
  --ocsv OCSV           output .csv file to save scores. By default the CSV name is inferred from the input file name, with --suffix input added. (default: None)
  --suffix SUFFIX       suffix added to output PDB file (default: None)
  --cautious            Cautious mode. If output CSV exists, then it will not run that prediction again. (default: False)
  --exclude_common_ligands
                        All common solvents and crystallography additivies will be excluded from the prediction.
                        List of residues was obtained from AlphaFold3 supplementary data (DOI: 10.1038/s41586-024-07487-w). Useful when predicting directly any crystal structures.
                        (default: False)
  --predict_multi       All allowed ligands in input will be predicted and scored. fixed_ligand and predict_ligand inputs are respected. (default: False)
  --fixed_ligand FIXED_LIGAND [FIXED_LIGAND ...]
                        Ligand <name3> or <name3-resno> or <chain-name3-resno> that will remain fixed. (default: None)
  --predict_ligand PREDICT_LIGAND [PREDICT_LIGAND ...]
                        Ligand <name3> or <name3-resno> or <chain-name3-resno> that will be predicted. All other ligands will be fixed. (default: None)
  --target_res TARGET_RES
                        Protein residue <chain-resno> or <chain-name3-resno> that will be used as crop center. Required when input has no ligands. (default: None)
  --fixed_ligand_noise FIXED_LIGAND_NOISE
                        Noise added to fixed ligand coordinates. Default is the same as backbone atom `sigma_bb` in the model params. (default: None)
  --weights WEIGHTS     Weights file (pytorch .pt file). (default: weights/PLACER_model_1.pt)
  --rerank {prmsd,plddt,plddt_pde}
                        Output CSV and PDB models files are ranked from best to worst based on one of the input metrics: prmsd, plddt, plddt_pde.
                        Prmsd is sorted in ascending order; plddt and plddt_pde in descending order.
                        The model numbers that are printed to screen while the script runs no longer apply. (default: None)
  --bonds BONDS [BONDS ...]
                        put a bond between two atoms, e.g. "A-42-ALA-CB:B-173-JRP-CL:<bondlen>", as space-separated list (default: None)
  --mutate MUTATE [MUTATE ...]
                        mutate certain positions, e.g. "5A:TRP" or "5A:TRP 6A:GLY" (default: None)
  --crop_centers CROP_CENTERS [CROP_CENTERS ...]
                        Atom names that will be used as CROP centers. This centers the crop to a particular part of the pocket, but the ligands are still corrupted based on their input coordinates.
                        Used for refining where the cropped sphere is. This DOES NOT affect which atoms/ligands are selected for prediction. Use --predict_ligand ... for that.
                        One atom will be picked randomly from the provided set.
                        XYZ coordinate input available in the API. Example: "B-200-HEM-FE B-200-HEM-O1" (default: None)
  --corruption_centers CORRUPTION_CENTERS [CORRUPTION_CENTERS ...]
                        Atom names that will be used as corruption centers. Allows sampling the ligand around in the whole protein.
                        One will be picked randomly from the provided set. Must provide at least as many centers as there are ligands in the input.
                        XYZ coordinate input available in the API. Example: "B-200-HEM-FE B-200-HEM-O1" (default: None)
  --residue_json RESIDUE_JSON
                        JSON file that specifies any custom residues used in the PDB, or used with --mutate. These are added to the internal CCD library.
                        JSON format: {name3: {'sdf': <contents of SDF file as string>,
                                              'atom_id': [atom names],
                                              'leaving': [True/False for whether this atom is deleted when part of polymer],
                                              'pdbx_align': [int,...]}} (default: None)
  --ligand_file LIGAND_FILE [LIGAND_FILE ...]
                        SDF or MOL2 file of the ligand(s). (Input format: XXX:ligand1.sdf YYY:ligand2.mol2) ZZZ:CCD
                        Used for refining the atom typing and connectivity in the ligand structures. Coordinates are still parsed form the input PDB/mmCIF.
                        If ligand exists in CCD then ZZZ:CCD is a special input that enables reading the ligand in from an internal CCD ligands database. (default: None)
  --ignore_ligand_hydrogens
                        Affects --ligand_file. Ignores hydrogen atoms that are defined in the PDB and SDF/MOL2 files, and will not throw errors if the protonation states are different.
                        Hydrogen atoms are not predicted with PLACER anyway. (default: False)
  --use_sm              make predictions with the small molecule (holo - turned on by default) (default: True)
  --no-use_sm           make predictions w/o the small molecule (apo) (default: True)




```

<a id="examples"></a>
## Examples
Some example commands from `examples/commandline_examples.sh`:
```
# predicting the binding of an inhibitor in a P450 pocket, while keeping heme fixed (heme is fixed automatically if it's not in --predict_ligand input)
python ../run_PLACER.py --ifile inputs/4dtz.cif --odir example_out_CLI --rerank prmsd --suffix D-LDP-501 -n 10 --predict_ligand D-LDP-501

# predicting the binding of an inhibitor and heme in a P450 pocket, docking and scoring two ligands simultaneously
python ../run_PLACER.py --ifile inputs/4dtz.cif --odir example_out_CLI --rerank prmsd --suffix LDP-HEM -n 10 --predict_ligand D-LDP-501 C-HEM-500 --predict_multi

# predicting heme in denovo protein
python ../run_PLACER.py --ifile inputs/dnHEM1.pdb --odir example_out_CLI --rerank prmsd -n 10 --ligand_file HEM:ligands/HEM.mol2

# predicting sidechains in apo denovo protein, defining crop center to a residue
python ../run_PLACER.py --ifile inputs/dnHEM1_apo.pdb --odir example_out_CLI --suffix A149 -n 10 --target_res A-149

# Mutating a residue to a non-canonical, loading that non-canonical into residue database from a JSON file. Existing ligand is omitted from the prediction
python ../run_PLACER.py --ifile inputs/denovo_SER_hydrolase.pdb --odir example_out_CLI --suffix 75I -n 10 --mutate 128A:75I --residue_json ligands/75I.json --no-use_sm
```

<a id="api_usage"></a>
## Python API usage

PLACER is available as a Python-importable module, although currently not pip-installable:
```
import sys
sys.path.append("<PLACER_path>")
import PLACER

pdbfile = "examples/dnHEM1.pdb"
# pdbstring = open(pdbfile, "r").read()

placer = PLACER.PLACER()  # loads the model from default checkpoint

pl_input = PLACER.PLACERinput()
pl_input.pdb( pdbfile )  # also works with pdbstring
pl_input.name( “heme_test” )
pl_input.ligand_reference( {"HEM": f"examples/ligands/HEM.mol2"} )  # sdf/mol2 file of the ligand

# Run the predictions:
outputs = placer.run( pl_input, 50 )  # dict output

```
Example Python script (`examples/API_examples.py`) and Jupyter notebook (`examples/API_examples_notebook.ipynb`) are available to explore the different API options.

<a id="faq"></a>
## FAQ and troubleshooting

Some more common questions and problems related to running PLACER:

**1) How many models should I generate?<br>**
    It depends. For sidechain conformation analysis, 50 is usually enough. For ligand docking, 50-100 can be enough, but if there aren't enough high confidence models then >200 might be necessary.

**2) What do the different scores mean?<br>**
    `fape` - all-atom FAPE loss<br>
    `lddt` - actual all-atom lDDT score between the generated model and the reference input structure (not a prediction)<br>
    `rmsd` - RMSD of the ligand atoms between the input (ground truth) and the predicted structure. Measures the accuracy of the docking position prediction.<br>
    `kabsch` - superimposed RMSD of the ligand atoms between input and predicted structure. Measures the accuracy of the ligand conformation prediction.<br>
    `prmsd` - RMS of predicted deviations of atomic positions of the ligand; predicted uncertainties in atomic positions.<br>
    `plddt` - predicted lDDT score averaged over ligand atoms, originates from the 1D track of the network.<br>
    `plddt_pde` - predicted lDDT score averaged over ligand atoms, originates from the 2D track of the network.<br>

**3) What confidence score should I use?<br>**
    We have seen "prmsd" offer the best performance in docking tasks. A good "prmsd" score depends on the complexity of the molecule. prmsd < 2.0 is trustworthy, but even < 4.0 can be acceptable if that's as good as it gets and the plddt and plddt_pde scores of that prediction are good (> 0.8).

**4) How should I use PLACER for ligand docking?<br>**
    It is best to perform docking analysis based on the highest confidence models. For example looking at 10% highest confidence predictions when ranked by prmsd.

**5) How can I get the confidences of each residue in the protein?<br>**
    The generated PDB file `filename_model.pdb` contains the "prmsd" scores of each atom in each model in the b-factor column.

**6) Can I use a larger crop size (> 600 atoms)?**<br>
    The model was not trained with more atoms, so unexpected behavior and reduced confidence is very likely if one tried.

**7) Can I directly use outputs of other structure prediction methods for PLACER analysis?**<br>
    While it's technically possible, we found that there can be challenges because of missing hydrogens, especially on ligands.
    mmCIF files produced by other methods can also be problematic because of differences in formatting compared to the mmCIF files from RCSB.
    Also, make sure you are not in violation of the terms and conditions of any of these methods when using their outputs for docking.

**8) Can I use SMILES as ligand input?**<br>
    No, unfortunately, currently the docked ligand must be present in the input PDB/mmCIF file. It is currently not possible to start a PLACER job by taking an apo-protein and giving it a separate ligand from a SMILES or SDF file.

**9) Can I perform global docking with PLACER?**<br>
    Technically yes. One can use the `--corruption_centers ...` flag (or corresponding method in the API) to sample potential binding pocket locations around the protein. This is not ideal, and we should also note that PLACER was not explicitly developed for this task.

**10) Why are my aromatic rings not planar?**<br>
    A potential cause is that PLACER assigned wrong atom hybridization and bond features to some of the atoms in the aromatic ring. This usually happens if PLACER predictions are run on PDB files without providing any additional information about the ligand structure. To remedy this, users can provide a MOL2/SDF file of that ligand with the flag `--ligand_file LIG:ligand.sdf`, and PLACER will read in correct information about the hybridization and bonding situation in that ligand. Occasionally the model also jsut produces distorted molecules, but the confidence scores tend to be worse in that case too.

**11) How can I predict with non-canonical amino acids?**<br>
    Non-canonical amino acids can be introduced to PLACER using a residue JSON file. The JSON file is provided to the commandline script with `--residue_json <file.json>`. In the API one needs to provide the corresponding dictionary with `pl_input.add_custom_residues(dict)`.
    The JSON file needs to contain a dictionary with the following contents:<br>
```
    {name3:   # 3-letter code of that residue
       {'sdf': <contents of SDF file as string>,   # SDF file with correct bonding and chirality info, as a single string
        'atom_id': [atom names],  # atom names as they appear on the PDB file, in the same order as atoms in the SDF
        'leaving': [True/False for whether this atom is deleted when part of polymer],  # True for backbone atoms that get deleted, such as backbone amide NH and backbone carboxylate OXT
        'pdbx_align': [int,...]   # list with a length of number of atoms. Can just be zeroes. The exact values no not affect PLACER as it was used by the mmCIF writer.
       }}
```

**12) How long do PLACER predictions take?**<br>
    One model of an average ligand-protein complex will be predicted on a GPU within 1-3 seconds, depending on how powerful the GPU is. On 1 CPU core it will take around 7 minutes, and on 8 CPU cores it will take around 1 minute. Ligands with larger number of automorphisms (many symmetric groups) will take longer to predict.

<a id="disclaimer"></a>
## Disclaimer
PLACER ships with a copy of a third party software, [pdbx](https://github.com/soedinglab/pdbx), which is used in the structure parser module.<br>
PLACER contains a copy of custom-formatted CCD (Chemical Components Dictionary) dataset, used by the structure parser.

