# Welcome to WHOLISTIC-registration
**A fast, accurate, and non-rigid image registration method for Whole-Body cellular activity Imaging.**


## Introduction
WHOLISTIC Registration is a method designed to correct non-rigid motion caused by skeletal and smooth muscle contractions. This enables precise cellular activity analysis and motion analysis for fluorescent data. The method utilizes patch-wise iterative modified optical flow with an image pyramid to achieve high flexibility and robustness.

Below are examples showcasing the results of WBI Registration:

-**Input (Left)**: Raw video

-**Output (Right)**: Motion-corrected video

<p align="center">
  <b>
    Example #1
  </b>
</p>

https://github.com/user-attachments/assets/871dfb15-49a5-47de-8b18-878880605e74

<p align="center">
  <b>
    Example #2
  </b>
</p>


https://github.com/user-attachments/assets/5fb45eca-02a1-4226-8208-2f6dbd6171a3

### Advantage
- **Accurate non-rigid registration**: Delivers precise motion correction.
- **Fast and parallelizable**: Optimized for GPU acceleration.
- **Flexible masking**: Allows users to ignore unwanted regions.
- **Robust to noise**: Performs well even with noisy fluorescent data.

### Suitable Data for WBI Registration
WBI Registration assumes that the template image and the moving image have similar intensities (or change gradually). Thus, At least one channel of your data should avoid fast-changing activities, such as Calcium activity signals, to ensure optimal results.


## Requirements
### OS Requirements
This package is supported for *Linux* and *Windows*. The package has been tested on the following systems:
+ Linux: Ubuntu 24.04


### Hardware Requirements
- A discrete GPU with sufficient memory is recommended for acceleration.


## License

This software is licensed under the **BSD 3-Clause License**.

Copyright © 2024 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice
2. Redistributions in binary form must reproduce the above copyright notice in the documentation
3. Neither the name of HHMI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission

See the [LICENSE](LICENSE) file for the full license text.

## Acknowledgments

This project was developed at [Janelia Research Campus](https://www.janelia.org/), Howard Hughes Medical Institute.

## Citation

If you use this software in your research, please cite:

```
[Citation information to be added]
```
