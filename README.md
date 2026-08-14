# Drift-Sense: Navigation-Error Recovery

## Overview

Drift-Sense is a computer-vision-based localization system designed to recover the position of a semiconductor reference pattern inside a larger search image.

The system is designed for semiconductor inspection scenarios where repetitive structures can cause conventional template matching to select an incorrect but visually similar location.

The project contains:

- A configurable dataset generator
- A localization inference script
- Ground-truth generation
- Image augmentation and degradation
- Localization evaluation
- Error analysis and failure visualization

---

## Project Pipeline

Reference Image + Search Image
          |
          v
   Image preprocessing
          |
          v
 Multi-scale / multi-angle localization
          |
          v
 Candidate verification
          |
          v
 Predicted center (x, y)
          |
          v
 Compare with ground truth
          |
          v
 Localization error

---

## Repository Structure

```text
drift-sense/
│
├── dataset_generator.py
├── localization_inference.py
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── reference/
│   ├── search/
│   ├── ground_truth.csv
│   └── ...
│
└── ...
