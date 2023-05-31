## Methods associated with an MPhys project titled "Investigating the Effect of Slice Thickness on Ultrasound Imaging Performance"

### Code used to generate and analyse data for the 20 credit project undertaken in final year. The final report is given in thesis.pdf and the abstract is as follows:

*The effect of slice thickness on ultrasound image quality is not well quantified. The resolution integral is an established technique for calculating a single figure-of-merit that combines imaging performance over an entire ultrasound beam. Using a commercial slice thickness phantom, continuous slice thickness versus depth profiles were recorded for 10 transducers and the resolution integral was adapted to quantify imaging performance solely in the elevation plane. Slice thickness was identified as having a significant impact on ultrasound imaging performance but no clear relationship was found between the calculated resolution integrals, likely due to confounding factors from the lateral plane. There was a strong linear relationship between the sizes of the overall and elevational focal regions, while the typical resolutions were correlated but also affected by confounding factors.*

![The performance of the code in resolution_integral.py versus an NHS spreadsheet for calculating the resolution integral from Edinburgh Pipe Phantom measurements.](nice_results/codehists.png)

These histograms show the performance of the code in resolution_integral.py versus an NHS spreadsheet for calculating the resolution integral from Edinburgh Pipe Phantom measurements. The code is much faster and still has high accuracy in most cases.

Description of code in `sample/`:
* `resolution_integral.py` calculates the resolution integral from measurements made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
* `correlations.py` looks for correlations between any two variables (e.g. frequency and characteristic resolution).
* `multiplot.py` useful functions for plotting slice thickness profiles together.
* `roi.py` used to find the coordinates of the Region of Interest for each ultrasound video and crop out the rest.
* `Rs_plot.py` used to plot characteristic resolution versus depth of field for all transducers studied here as well as those from other studies.
* `slice_thickness.py` finds the slice thickness versus depth from a video.
* `ST_uncertainties.py` calculate the uncertainty in slice thickness values.
* `test_autores.py` plot the code performance histograms.
* `utils` function for loading data.
* `videos.py` called by `slice_thickness.py` to analyse the videos.

![Example of a slice thickness depth profile extracted from a video, also shows the smoothing and trimming process applied to make the data useable.](nice_results/ST23.png)
Example of a slice thickness depth profile extracted from a video, also shows the smoothing and trimming process applied to make the data useable.