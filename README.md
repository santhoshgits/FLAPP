# FLAPP (Fast Local Alignment of Protein Pockets) is a system-compiled python program for doing binding site alignment in large scale.
## FLAPP approximately takes 10 milli second to align two binding sites on a single CPU cores. Our script also has support for multi processing. When N=12, FLAPP takes just 1 milli second to align site pairs.
**FLAPP require two scripts, one for converting the input pdb into binary format and another to establish the actual alignment**

## Instruction to install FLAPP
The algorithm has been implemented in Python-3.9 (Anaconda distribution). It is therefore recommended to install the same version of python in your system to run our code.

Once Python-3.9 is installed, we then have to install additional libraries such as Numba and NumPy. To enable easy installation, A YAML file called environments.yml is provided which will invoke a seprate environments to run FLAPP safely.

****

**Use the below command to create environments**
```markdown
 
1. conda env create -f environment.yml
The above command will create an environment called 'FLAPP' and install all pre-requisite libararies in your machine.

2. conda activate FLAPP
To activate and load all modules required to run FLAPP. 

3. conda deactivate.
To deactivate FLAPP.
```
---


# Steps for Running FLAPP

FLAPP requires PDB files of all input binding sites in a single directory. For the sake of explanation, i have added 200 ATP binding sites in the folder 'BindingSites'.

FLAPP next requires a tab separated file that contains what pairs of site have to aligned. Please refer to 'Pairs.txt' for the structure. Make sure the inputs present in Pairs.txt is also present in the 'BindingSites' folder.

An 200-against-200 sites will generate 40,000 pairwise comparisons as seen in Pairs.txt.

## STEP-1: Converting input binding site in to python objects (byte stream)
### python CreateVector.py Argument-1 Argument-2

Usage: python CreateVectors.py BindingSites SiteVector  

1. **Argument-1 -** Provide the folder name that contain our binding sites as the first argument to this script

2. **Argument-2 -** Specify the folder that will store the binarised output. 


## STEP-2: Running the actual FLAPP program
### python FLAPP.py Argunemt-1 Argument-2 Argument-3 Argument-4

Usage: python FLAPP.py SiteVector Pairs.txt Outfile.txt 4
  
1. **Argument-1 -** Provide the name of the output folder from step-1

2. **Argument-2 -** Specify the Pairs.txt file. (Tab Delimited)

3. **Argument-3 -** Specify the file name to store the result

4. **Argument-4 -** Specify the number of CPU cores to utilize for the run


---
