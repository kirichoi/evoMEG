# evoMEG

Evolutionary Algorithm-based Model Ensemble Generation

COPYRIGHT 2023 Kiri Choi

Evolutionary Algorithm-based Model Ensemble Generation (evoMEG) is a meta-modeling algorithm that generates an ensemble of mechanistic biochemical reaction network models from the output perturbation studies such as scaled concentration control coefficients.

## How to use

`python main.py -s <settingFile> -m <modelFile> -c <cacheDirectory> --help`

Configure `settings.txt` file or modify `main.py` directory to change the parameters. See `./settings` folder for examples. 
Custom model can be loaded in using `-m` or `--model`. 
Cached output can be used to rerun the algorithm using `-c` or `--cache`.
Various example cases are available for a test. Try adding `modelType: <test model>` to try.
Currently, we support the following test cases:

`FFL_m/FFL_r`: irreversible/reversible feedforward loop
`Linear_m/Linear_r`: irreversible/reversible linear chain
`Nested_m/Nested_r`: irreversible/reversible cycles
`Branched_m/Branched_r`: irreversible/reversible branching pathways
`Feedback_m/Feedback_r`: irreversible/reversible feedback loop

## Settings

### Input
MODEL_INPUT: Path to a custom model (default: None)
READ_SETTINGS: Path to preconfigured settings (default: None)
CACHED_DIR: Path to a previous output (default: None)

### General
ens_size: Size of output ensemble
pass_size: Number of models used for recombination (default: int(0.1*ens_size))
top_p: Top percentage of population to track (default: 0.1)
maxIter_init: Maximum iteration allowed for initialization (default: 10000)
maxIter_gen: Maximum iteration allowed for random generation (default: 200)
maxIter_mut: Maximum iteration allowed for mutation (default: 200)
recomb: Recombination probability (default: 0.3)
conservedMoiety: Set conserved moiety (default: False)
checkCorrectStoichiometry: When testing, check if the correst stoichiometry was recovered (default: True) 
prune: Prune unnecessary boundary species at the end (default: True)

### Termination criterion
n_gen: Maximum number of generations
gen_static: Number of generations w/o improvement
thres_avg: Threshold average distance
thres_median: Threshold median distance
thres_shortest: Threshold shortest distance
thres_top: Threshold top p-percent smallest distance
max_run_time: Maximum run time allowed in minutes

### Optimizer
optiMaxIter: Maximum iteration allowed (default: 1000)
optiTol: Optimizer tolerance (default: 1)
optiPolish: Allow polishing parameters for optimizer (default: False)
refine: Run optimization at the end for better fitness representation (default: False)
refineTol: Tolerance for additional optimization at the end (default: 0.01)

### Reaction
kineticType: Reaction kinetics - 'default', 'mass-action' (default: 'default')

### RNG and noise
r_seed: Seed
NOISE: Flag to add Gaussian noise to the input
ABS_NOISE_STD: Standard deviation of absolute noise
REL_NOISE_STD: Standard deviation of relative noise

### Plotting
SHOW_PLOT: Flag to visualize plot
SAVE_PLOT: Flag to save figures

### Export
EXPORT_ALL_MODELS: Flag to collect all models in the ensemble
EXPORT_OUTPUT: Flag to save collected models
EXPORT_SETTINGS: Flag to save current settings
EXPORT_CACHE: Flag to save model components for caching
EXPORT_PATH: Path to save the output
EXPORT_OVERWRITE: Overwrite the contents if the folder exists
EXPORT_FORCE_MODELNAMES: Create folders based on model names

## Acknowledgements

This work was supported by the National Institute of General Medical Sciences of the National Institutes of Health under awards R01-GM081070, R01-GM123032 and KIAS individual grant CG077002. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.