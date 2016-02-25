# README #

Original file is from [OpenCV samples](https://github.com/Itseez/opencv/blob/master/samples/python/deconvolution.py).
Changes in this repository enabled processing of color images (treated as three independent channels, finally merged into one RGB image) instead of only monochromatic.

### About ###

This code performs Wiener deconvolution in order to inverse the impact of image focus blur or motion blur. In order to do that OpenCV and NumPy is used.

### Examples ###

Focus blur reduction:
![example 1: focus blur](https://bytebucket.org/michal_229/dft-wiener-deconvolution-with-psf/raw/dab66dbc1ea6d823507b38e4a49cdfa3f1e997ec/case%201%20-%20focus%20blur.png "focus blur")

Motion blur reduction:
![example 2: motion blur](https://bytebucket.org/michal_229/dft-wiener-deconvolution-with-psf/raw/dab66dbc1ea6d823507b38e4a49cdfa3f1e997ec/case%202%20-%20motion%20blur.png "motion blur")