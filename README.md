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
