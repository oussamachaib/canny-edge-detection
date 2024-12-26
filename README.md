# canny-edge-detection
 
|       ![plot](./demo_images/intro.png)        |
| :-------------------------------------------: |
| *Personal implementation of Canny's operator* |

 Python implementation (coded entirely in numpy) of the ["A Computational Approach to Edge Detection"](https://ieeexplore.ieee.org/document/4767851) (1986) - Canny's operator for edge detection from digital images.

The algorithm takes a grayscale image as input, in addition to parameters pertaining to the convolutional kernel parameters (Gaussian standard deviation and width), and hysteresis thresholds, then output a binary image corresponding to object edges in the image. They are defined a local maxima of gradient magnitudes in the image.

In the Jupyter Notebook, I demonstrate the performance of my implementation on pictures of four flower species.

While this was done for learning purposes, it might be useful for those trying to build an intuition about the method (and convolutional kernels and operations, in general).

##### Contents
---------------
```canny_operator.py```: Python implementation of the algorithm.

```demo.ipynb```: Demonstration.