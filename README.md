# Drift-Sense: Navigation-Error Recovery

## Overview

Drift-Sense is a computer-vision-based localization system designed to recover the position of a semiconductor reference pattern inside a larger search image.

The project addresses navigation errors that can occur during semiconductor wafer inspection when repetitive structures make multiple regions of an image appear visually similar. Conventional template matching may therefore select an incorrect location.

Drift-Sense combines multi-scale and multi-angle image matching with candidate verification to improve localization robustness under image degradation, geometric variation, and periodic semiconductor layouts.

---

# Problem Statement

In semiconductor inspection, a small reference pattern may need to be located inside a much larger SEM search image.

The main challenge is that semiconductor layouts often contain highly repetitive structures. A simple template-matching algorithm can therefore find a visually similar but incorrect region.

Drift-Sense aims to recover the correct `(x, y)` location of the reference pattern while handling:

- Noise
- Blur
- Rotation
- Scale changes
- Contrast changes
- Brightness variation
- SEM edge brightening
- Periodic background structures

---

# Key Features

- Computer-vision-based localization
- Supports DRAM and FinFET architecture styles
- Configurable synthetic dataset generation
- Ground-truth coordinate generation
- Multi-scale localization
- Multi-angle localization
- Candidate generation
- Candidate verification
- Independent reference/search image degradation
- Localization evaluation
- Failure-case visualization
- Standalone inference script for external evaluation

---

# Project Pipeline

```text
Reference Image + Search Image
            |
            v
     Image Preprocessing
            |
            v
 Multi-scale / Multi-angle Search
            |
            v
     Candidate Generation
            |
            v
    Candidate Verification
            |
            v
      Best Candidate
            |
            v
    Predicted Center (x,y)
            |
            v
      Ground-truth Comparison
            |
            v
     Localization Error
Repository Structure
drift-sense/
│
├── dataset_generator.py
├── localization_inference.py
├── localization.py
├── general_real_sem.py
├── final_analysis.py
├── failure_visualization.py
│
├── README.md
├── requirements.txt
├── references.md
│
└── references/
Project Files
File	Purpose
dataset_generator.py	Generates reference/search image pairs and ground truth
localization_inference.py	Standalone inference script for predicting (x,y)
localization.py	Main localization and evaluation implementation
general_real_sem.py	Real SEM dataset preparation
final_analysis.py	Quantitative result and error analysis
failure_visualization.py	Visualizes worst localization cases
requirements.txt	Python dependencies
references.md	Supporting references
README.md	Project documentation
Technology Stack
Programming Language

Python 3.x

Libraries
OpenCV
NumPy
Pandas
Matplotlib

The complete package list is provided in:

requirements.txt
Installation

Clone the repository:

git clone https://github.com/manvi25ecstu009-art/drift-sense.git
cd drift-sense

Install the required packages:

pip install -r requirements.txt
Dataset Generator

The project includes a configurable dataset generator:

dataset_generator.py

The generator creates:

Reference images
Search images
Ground-truth coordinates
Preview images

The generator supports two semiconductor architecture styles:

DRAM
FinFET
Dataset Generator Usage
DRAM
python dataset_generator.py --architecture DRAM --pairs 10 --output dataset
FinFET
python dataset_generator.py --architecture FinFET --pairs 10 --output dataset

The generated directory has the following structure:

dataset/
├── reference/
├── search/
├── previews/
└── ground_truth.csv

The ground_truth.csv file records the true center coordinates of the reference pattern in each generated search image.

Dataset Generator Parameters

The dataset generator accepts the following parameters:

--architecture
--pairs
--output

Example:

python dataset_generator.py --architecture DRAM --pairs 2 --output test_dataset

Example:

python dataset_generator.py --architecture FinFET --pairs 2 --output test_dataset_finFET

The generated ground-truth file records the correct reference location for each image pair.

Dataset Variations

The dataset generator introduces several variations to simulate challenging semiconductor inspection conditions.

The implemented variations include:

Noise
Blur
Rotation
Scaling
Contrast variation
Brightness variation
SEM edge brightening
Periodic background structures
Independent reference/search degradation

The reference and search images can receive different degradation conditions, making the localization task more challenging.

Supporting references for the augmentation and imaging choices are provided in:

references.md

Additional supporting material is available in:

references/
Localization Method

Drift-Sense uses a classical computer-vision approach rather than a deep-learning model.

The localization process consists of:

Image preprocessing
Multi-scale search
Multi-angle matching
Candidate generation
Candidate verification
Candidate scoring
Selection of the best candidate
Prediction of the reference center (x, y)

The candidate verification stage is intended to reduce false matches caused by repetitive semiconductor layouts.

Standalone Inference Script

The most important file for external evaluation is:

localization_inference.py

The script accepts:

Reference image path
Search image path

and outputs the predicted center coordinate (x, y).

The script is designed to run without manual editing before execution.

Inference Usage

Run:

python localization_inference.py <reference_image> <search_image>

Example:

python localization_inference.py dataset/reference/reference_001.png dataset/search/search_001.png

Example output:

422 688

This represents:

x = 422
y = 688

The output is the predicted center coordinate of the reference pattern within the search image.

Testing the Inference Script

A complete test can be performed by first generating a small dataset:

python dataset_generator.py --architecture DRAM --pairs 2 --output test_dataset

Then run:

python localization_inference.py test_dataset/reference/reference_001.png test_dataset/search/search_001.png

The script should return a single (x, y) coordinate.

The same procedure can also be performed using FinFET:

python dataset_generator.py --architecture FinFET --pairs 2 --output test_dataset_finFET

Then run:

python localization_inference.py test_dataset_finFET/reference/reference_001.png test_dataset_finFET/search/search_001.png
Evaluation

The current system was evaluated on 30 image pairs.

Localization error is calculated using the Euclidean distance between the predicted center and the ground-truth center.

Current Results
Metric	Result
Number of test pairs	30
Mean localization error	43.48 px
Median localization error	11.29 px
Minimum error	2.00 px
Maximum error	163.89 px
Within 5 px	26.7%
Within 10 px	46.7%
Within 20 px	60.0%

These values represent the current measured performance on the project's 30-pair evaluation dataset.

The results are reported honestly, including failure cases, rather than excluding unsuccessful predictions.

Error Analysis

The project contains scripts for quantitative and visual error analysis.

Final Analysis

Run:

python final_analysis.py

The analysis produces:

Error per pair
Error distribution
Accuracy threshold analysis
Best cases
Worst cases
Final summary

The analysis identifies both successful and unsuccessful localization cases.

Failure Visualization

The project contains a failure visualization script:

failure_visualization.py

Run:

python failure_visualization.py

The script selects the worst localization cases and creates visualizations showing:

Ground-truth position
Predicted position
Error direction
Localization error in pixels

These visualizations are used to understand why the algorithm fails on difficult repetitive structures.

Reproducibility

A reviewer can reproduce the basic pipeline using the following steps.

Step 1 — Clone the Repository
git clone https://github.com/manvi25ecstu009-art/drift-sense.git
cd drift-sense
Step 2 — Install Dependencies
pip install -r requirements.txt
Step 3 — Generate a Test Dataset
python dataset_generator.py --architecture DRAM --pairs 2 --output test_dataset
Step 4 — Run Localization
python localization_inference.py test_dataset/reference/reference_001.png test_dataset/search/search_001.png
Step 5 — Obtain Prediction

The inference script outputs a single coordinate:

x y

This coordinate represents the predicted center of the reference pattern in the search image.

Hardware and Computational Requirements

The project is implemented using Python and classical computer-vision methods.

No deep-learning model or model weights are required for inference.

The system can therefore be executed on a standard computer capable of running Python and OpenCV.

Deep Learning

Drift-Sense currently does not use a deep-learning model.

Therefore:

No .pt model weights are required.
No .h5 model weights are required.
No ONNX model is required.
No training script is required.

The localization system uses classical computer-vision techniques.

Innovation and Uniqueness

The project focuses on navigation-error recovery in repetitive semiconductor layouts.

The main design elements are:

1. Architecture-Aware Dataset Generation

The generator supports both DRAM and FinFET-style structures.

2. Independent Image Degradation

Reference and search images can undergo different degradation conditions rather than assuming identical image quality.

3. Multi-Scale Localization

The search considers multiple scales to handle differences in pattern size.

4. Multi-Angle Localization

Multiple orientations are considered to improve robustness to rotation.

5. Candidate Verification

Candidate matches are additionally evaluated using structural information to reduce false matches caused by repetitive patterns.

6. Explicit Failure Analysis

The project does not only report successful cases.

Worst-case predictions are also visualized and analyzed to identify limitations of the current method.

Limitations

The current evaluation shows that repetitive semiconductor structures remain challenging.

Some cases produce large localization errors because visually similar periodic regions can receive high matching scores.

Current results show:

46.7% of predictions within 10 pixels
60.0% of predictions within 20 pixels

This indicates that further candidate disambiguation and structural reasoning are areas for future improvement.

Future Improvements

Potential improvements include:

Stronger periodic-pattern disambiguation
More robust structural descriptors
Improved candidate ranking
Larger and more diverse SEM datasets
More realistic semiconductor architecture simulation
Hybrid classical + learning-based localization
Additional validation on unseen SEM images
References

References supporting the image-processing, augmentation, semiconductor imaging, and computer-vision choices are provided in:

references.md

Additional supporting material is available in:

references/
Project Status

The current repository provides:

DRAM dataset generation
FinFET dataset generation
Synthetic image degradation
Ground-truth generation
Multi-scale localization
Multi-angle localization
Candidate verification
Standalone inference
Quantitative evaluation
Failure-case analysis
Reproducible setup instructions
GitHub Repository

https://github.com/manvi25ecstu009-art/drift-sense.git

Quick Start

After cloning the repository, run:

git clone https://github.com/manvi25ecstu009-art/drift-sense.git
cd drift-sense
pip install -r requirements.txt
python dataset_generator.py --architecture DRAM --pairs 2 --output test_dataset
python localization_inference.py test_dataset/reference/reference_001.png test_dataset/search/search_001.png

The final command should output one predicted coordinate:

x y

This coordinate is the predicted center of the reference pattern in the search image.
