# Drift-Sense: Navigation-Error Recovery

## Overview

Drift-Sense is a computer-vision based localization system designed to locate a small reference pattern inside a larger SEM search image.

The project focuses on navigation-error recovery in semiconductor wafer inspection, where repetitive structures can make simple template matching unreliable.

The system uses OpenCV-based image matching with:

- Multi-scale template matching
- Multi-angle matching
- Edge-based matching
- Image preprocessing
- Noise and image-degradation augmentation
- Automatic localization error evaluation

---

## Problem

In semiconductor inspection, a reference pattern may need to be located inside a larger search image.

The search image can contain:

- Noise
- Blur
- Rotation
- Scaling differences
- Contrast variations
- Brightness variations
- Strong edges
- Repetitive semiconductor structures

These effects can cause a conventional template-matching algorithm to select an incorrect but visually similar region.

Drift-Sense attempts to make localization more robust to these variations.

---

## Project Pipeline

```text
SEM Image
    |
    v
Reference Pattern + Search Image
    |
    v
Image Preprocessing
    |
    v
Multi-Scale Matching
    |
    v
Multi-Angle Matching
    |
    v
Edge-Based Matching
    |
    v
Best Candidate Selection
    |
    v
Predicted (x, y)