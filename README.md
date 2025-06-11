# Installation and Data Preparation

## Step 1: Install requirements

Requirements are listed in the requirements.txt file, and can be installed using pip (pip install -r requirements.txt).
To use ByteTrack, install the module (to the root directory) as described in the corresponding [repository](https://github.com/yhsmiley/bytetrack_realtime).

## Step 2: Prepare evaluation data

Download the evaluation data, and unpack the zipfile into the root directory of the repository.

# Reproduction of results from the BPM 2023 Forum Paper "Analytics Pipeline for Process Mining on Video Data"

Find the paper here: [https://link.springer.com/chapter/10.1007/978-3-031-41623-1_12](https://link.springer.com/chapter/10.1007/978-3-031-41623-1_12)
## Step 1: Internal evaluation

To reproduce internal validation, run [evaluation/cross_validation.py](evaluation/cross_validation.py) (with the abstraction smoothing parameter set to 20).
The conformance checking results are output to a .csv file in the same directory, and can be visualized using [evaluation/plot_evaluation.py](evaluation/plot_evaluation.py).

## Step 2: External evaluation

The event log used for external evaluation and additional visualizations are provided in the directory [case_study](case_study).
We used [Disco](https://fluxicon.com/disco/) to discover the models shown in the paper.

# "Describing behavior sequences of fattening pigs using process mining on video data and automated pig behavior recognition"

Find the paper here: [https://www.mdpi.com/2077-0472/13/8/1639](https://www.mdpi.com/2077-0472/13/8/1639)

## Reproduction of Results

Use the scripts in [evaluation](evaluation) to reproduce the figures. To reproduce the models, import the XES-formatted event log [case_study/clustered_log_10s.xes](case_study/clustered_log_10s.xes) into Disco. Note that this event log used a smoothing parameter of 10 in the event abstraction, as opposed to 20 in the BPM Forum paper.

## Supplementary Materials

Find the process models of all clusters in [supplementary_materials](supplementary_materials)

# Data availability

The training datasets, weights of finetuned models and inference results for the five days selected for the case study are available on Zenodo: https://doi.org/10.5281/zenodo.7763838  

# Licenses

This repository is licensed under the CC-BY 4.0 license as included in [LICENSE.txt](LICENSE.txt).

We include OC_Sort under the terms of the MIT license.
ByteTrack cannot be included directly here, because the implementation we used does not provide a suitable license.
