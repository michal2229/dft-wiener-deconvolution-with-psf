# README #

Original file is from [OpenCV samples](https://github.com/Itseez/opencv/blob/master/samples/python/deconvolution.py).

### About ###

This code performs Wiener deconvolution in order to inverse the impact of image focus blur or motion blur. In order to do that OpenCV and NumPy is used.

Changes in this repository enabled:

* processing of color images (treated as three independent channels, finally merged into one RGB image) instead of only monochromatic
* using custom PSF loaded from image file (specified in command line)

### Examples ###

Focus blur reduction:

![example 1: focus blur](https://raw.githubusercontent.com/michal2229/dft-wiener-deconvolution-with-psf/master/results/case%201%20-%20focus%20blur.png)

Motion blur reduction:

![example 2: motion blur](https://raw.githubusercontent.com/michal2229/dft-wiener-deconvolution-with-psf/master/results/case%202%20-%20motion%20blur.png)

Example custom kernel obtained from a photo containing motion trail of small, bright object:

![example 3: custom kernel](https://raw.githubusercontent.com/michal2229/dft-wiener-deconvolution-with-psf/master/kernel/kernel_IMG_20160511_024929_HDR%20(kopia).png)



