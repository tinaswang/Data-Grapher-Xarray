from Data import Data
import xarray as xr
import pandas as pd
import numpy as np

d = Data(data_file="Data Examples/HiResSANS_exp9_scan0030_0001.xml",
         center_file="Data Examples/HiResSANS_exp9_scan0006_0001.xml",
         background_file="Data Examples/HiResSANS_exp9_scan0038_0001.xml")
d.setup()
size_x = 192
size_y = 256
dim_x = d.size[0]
dim_y = d.size[1]
z = 10
offset_x = d.center[0]
offset_y = d.center[1]

# 2D image in 1D
iv, jv = np.mgrid[0:size_x, 0:size_y]
iv = iv.ravel()
jv = jv.ravel()

# x,y,z real coordinates vectors
xv = offset_x + iv*dim_x
yv = offset_y + jv*dim_y
zv = np.full_like(xv, z)

# Concatenate all of them
allv = np.column_stack((iv, jv, xv, yv, zv))

df = pd.DataFrame(data=allv, columns=['i', 'j', 'x', 'y', 'z'])
# Set a multi index
df.set_index(['i', 'j'], inplace=True)
