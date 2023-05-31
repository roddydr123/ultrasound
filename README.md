## Methods associated with an MPhys project titled "Investigating the Effect of Slice Thickness on Ultrasound Imaging Performance"

### Code used to generate and analyse data for the 20 credit project undertaken in final year. The final report is given in thesis.pdf and the abstract is as follows:

*The effect of slice thickness on ultrasound image quality is not well quantified. The resolution integral is an established technique for calculating a single figure-of-merit that combines imaging performance over an entire ultrasound beam. Using a commercial slice thickness phantom, continuous slice thickness versus depth profiles were recorded for 10 transducers and the resolution integral was adapted to quantify imaging performance solely in the elevation plane. Slice thickness was identified as having a significant impact on ultrasound imaging performance but no clear relationship was found between the calculated resolution integrals, likely due to confounding factors from the lateral plane. There was a strong linear relationship between the sizes of the overall and elevational focal regions, while the typical resolutions were correlated but also affected by confounding factors.*

![The performance of the code in .](nice_results/codehists.png)

Description of code in `sample/`:
* `auto_res_check.py` calculates the resolution integral from measurements made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
* `correlations.py` looks for correlations between any two variables (e.g. frequency and characteristic resolution).
* `multiplot.py` 