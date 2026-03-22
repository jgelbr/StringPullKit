# StringPullKit
Automated pipeline for kinematic analysis of mouse string-pulling behaviour. Combines a video preprocessing GUI, DeepLabCut pose estimation, and an analysis pipeline to extract detailed forelimb, head, postural, and sensorimotor metrics from string-pull recordings.

# Overview
The string-pull task is a sensitive measure of fine forelimb coordination and postural control in rodents. StringPullKit provides end-to-end tooling for:

* __Video preprocessing__ — segmenting, rotating, cropping and spatially calibrating recordings via a GUI
* __Pose estimation__ — running DeepLabCut inference across 6 body-part networks (forepaws, arm joints, torso, hindlimbs, ears, nose)
* __Kinematic analysis__ — automated phase detection (reach/withdraw cycles) and extraction of 50+ metrics
* __Batch processing__ — headless processing of entire cohorts from a single parameter file (WORK IN PROGRESS)

# Installation
 ```bash
git clone https://github.com/jgelbr/StringPullKit.git
cd StringPullKit
pip install -e .
```
To include DeepLabCut support (required for running DLC tracking from the GUI:
```bash
pip install -e ".[dlc]"
```
Note: DeepLabCut has its own environment requiremnts. Consult the DeepLabCut installation guide if you haven't set it up already. You can alternatively use StringPullKit's analysis pipeline on existing DLC output without installing DeepLabCut, provided the models correspond to those used in the toolkit.

# Quick Start
__Launch the GUI__
```bash
python -m stringpullkit
```

# Using the GUI: Workflow
1. __Load video__: File --> Load Video 

2. __Trim and segment__: scrub through video, or input target frames to isolate behavioural epochs. Use Set Start/Set End to define active pulling, add as segments.

3. __Rotate and crop__: align field of view, draw and confirm a crop rectangle around target area.
  
4. __Calibrate__: select Set Scale, draw a line of known length on the video to set scale (pixels --> mm)
  
5. __Run DLC__: run DLC tracking and export edited video segment. Produces DLC csvs and labelled videos that can be viewed in the GUI. If not using DLC, edited video can be exported.
  
6. __Run analysis__: computes all kinematic metrics and saves to `.h5` and `.xlsx`.

Additional options
* __Load DLC CSVs__: perform analysis through the GUI with pre-generated DLC csvs.
* __Update DLC configs__: input paths to DLC model config files
* __....__ other functionalities that I will write about eventually.

# Metrics
Metrics are computed per-phase (reach and withdraw) and per-session. Output is saved as both `.h5` (full hierarchichal data) and `.xlsx` (human readable summary).

See `metrics.md` for more details.

# DeepLabCut Models
Pose estimation used 6 ResNet-50 networks trained for 1,030,000 iterations on manually annotated frames from string-pull recordings. Config files for each network are managed via File --> Update DLC Config Paths in the GUI, and cached locally for reuse. 

Networks:

* `Hands` — left and right forepaws
* `Arms` — shoulder, elbow, wrist (both sides)
* `Torso` — shoulders, hips, body midline
* `Feet` — left and right hindpaws
* `Ears` — left and right ears
* `String` — nose + string reference point

# Requirements
* Python (version?)
* numpy, scipy, pandas, matplotlib, seaborn
* h5py, xlsxwriter
* opencv-python, Pillow
* deeplabcut (optional)
See `requirements.txt` for details.

# Licence
MIT Licence. See `Licence` for details. 

# Contact
For questions/issues, open a GitHub issue or contact juliana.gelber@mail.mcgill.ca

TODO:
requirements.txt

config files

metrics.md

additional functionalities sections

demo videos and data

batch processing 


