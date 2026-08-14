# DRIFT-SENSE References

## 1. SEM Image Noise

Timischl, F. (2015). "The Contrast-to-Noise Ratio for Image Quality Evaluation in Scanning Electron Microscopy." *Scanning*, 37(1), 54–62.

This reference supports the use of noise and signal-to-noise considerations when evaluating SEM images.

DOI: 10.1002/sca.21179


## 2. SEM Signal-to-Noise Ratio

Thong, J. T., Sim, K. S., & Phang, J. C. (2001). "Single-image signal-to-noise ratio estimation." *Scanning*, 23(5), 328–336.

This work discusses noise characteristics in SEM images and methods for estimating image signal-to-noise ratio.

DOI: 10.1002/sca.4950230506


## 3. SEM Noise Sources

"Effect of shot noise and secondary emission noise in scanning electron microscope images."

This reference discusses shot noise and secondary-emission noise as sources of degradation in SEM images.

DOI: 10.1002/sca.4950260106


## 4. Repetitive Pattern Matching

Fan, B., Wu, F., & Hu, Z. (2011). "Towards reliable matching of images containing repetitive patterns." *Pattern Recognition Letters*, 32(14), 1851–1859.

This reference supports the problem of local ambiguity caused by repetitive patterns and motivates using structural/geometric information in addition to simple local matching.

DOI: 10.1016/j.patrec.2011.07.029


## 5. Repeated Pattern Detection

Pritts, J., Chum, O., & Matas, J. (2014). "Detection, Rectification and Segmentation of Coplanar Repeated Patterns." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2973–2980.

This reference discusses geometric ambiguity in repeated patterns and methods for improving their detection and localization.


## 6. DRIFT-SENSE Augmentations

The DRIFT-SENSE dataset generator uses controlled variations including:

- Gaussian/image noise
- Blur
- Rotation
- Scaling
- Contrast variation
- Brightness variation
- SEM-style edge brightening

These augmentations are intended to model image-quality and acquisition variations relevant to testing localization robustness. The SEM noise references above motivate noise-related variations, while the repetitive-pattern references motivate testing localization under transformations that can increase matching ambiguity.

## 7. Project Method

DRIFT-SENSE uses classical computer-vision localization rather than a trained deep-learning model.

The localization pipeline uses template-based matching with multiple scales and rotations, together with structural/edge information. Ground-truth coordinates are used only for evaluation and are not provided to the localization algorithm during prediction.
