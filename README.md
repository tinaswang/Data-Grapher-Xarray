# Data Grapher Xarray

Version of Data Grapher that uses xarray

This program takes data from a beam data file, beam center file (optional),
and background file (optional). These initial files must be in XML. There is a
version of this program that does not use xarray[] (xarray.pydata.org).

##How to Use

###Setup:

Only data file:

`d = (data_file="data file location")
 d.setup()`

Only center and data files:

`d = (data_file="data file location",
      center_file = "center file location")
  d.setup()`

All three files:

`d = (data_file="data file location",
      center_file = "center file location",
      background_file="background file location")
  d.setup()`

###Graphing

####Radial Profile:
To graph the radial profile, all three files (center, data, background) must
have been provided initially.

  `d.display()`

####Heatmap:

To graph a [plotly](https://plot.ly/python/) heatmap of the difference
between the data and the background data (or the center file if no
background has been provided), at minimum, the center file have been provided.

  `d.display2d()`

##Files Included

*   Display.py: graphs the data

*   Operations.py: is a static class that does most of the calculation (radial
integration and center of mass)

*   Parser.py: turns the XML files into usable python dictionaries and can also
dump as JSON files and JSON strings.

*   Data.py: bundles the three other files together.
