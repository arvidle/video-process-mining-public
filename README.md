# Reproduction of our results

## Step 1: Install requirements

Requirements are listed in the requirements.txt file, and can be installed using pip (pip install -r requirements.txt).
To use ByteTrack, install the module (to the root directory) as described in the corresponding [repository](https://github.com/yhsmiley/bytetrack_realtime).

## Step 2: Prepare evaluation data

Download the evaluation data, and unpack the zipfile into the root directory of the repository.

## Step 3: Internal evaluation

To reproduce internal validation, run [evaluation/cross_validation.py](evaluation/cross_validation.py).
The conformance checking results are output to a .csv file in the same directory, and can be visualized using [evaluation/plot_evaluation.py](evaluation/plot_evaluation.py).

## Step 4: External evaluation

The event log used for external evaluation and additional visualizations are provided in the directory [case_study](case_study).
We used [Disco](https://fluxicon.com/disco/) to discover the models shown in the paper.

# Data availability

The training datasets, weights of finetuned models and inference results for the five days selected for the case study are available on Zenodo: https://doi.org/10.5281/zenodo.7763838  

# Licenses

This repository is licensed under the CC-BY 4.0 license as included in [LICENSE.txt](LICENSE.txt).

We include OC_Sort under the terms of the MIT license.
ByteTrack cannot be included directly here, because the implementation we used does not provide a suitable license.
