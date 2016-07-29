import numpy as np
from scipy import ndimage
from Data import Data
import xarray as xr
from numba import jit
import matplotlib.pyplot as plt

def get_center_of_mass(d):
    d.setup()
    center_data = d.center_data
    pixel_size_x, pixel_size_y = d.size
    data = center_data.values
    p = np.array([0, 0])
    x_units_centered = center_data.coords['x'].values
    y_units_centered = center_data.coords['y'].values
    p_new = np.array([x_units_centered.max()/2, x_units_centered.max()/2])
    while np.absolute(np.linalg.norm(p_new) - np.linalg.norm(p)) > 0.25:
        bool_mask = np.zeros((data.shape[0], data.shape[1]), dtype = bool)
        bool_mask = make_mask(bool_mask, data, x_units_centered, y_units_centered, p, p_new)
        masked_data = np.ma.masked_array(data, mask = bool_mask)
        x_total, y_total, total_intensity = calculate_intensity(masked_data, x_units_centered, y_units_centered, data)
        p = p_new
        p_new = np.array([x_total/total_intensity, y_total/total_intensity])
    # com = ndimage.center_of_mass(data)
    # point = (x_units_centered[int(com[1])]/pixel_size_x, y_units_centered[int(com[0])]/pixel_size_y)
    # print(point)
    plt.imshow(data)
    plt.scatter(p_new[0]/pixel_size_x, (p_new[1])/pixel_size_y, color = 'w', s = 50)
    print(p_new[0]/pixel_size_x, (p_new[1])/pixel_size_y)
    plt.show()
    # print(point)

@jit(nopython=True)
def make_mask(bool_mask, data, x_units_centered, y_units_centered, p, p_new):
    for j in range(len(data)):
        for i in range(len(data[j])):
                if (np.absolute(y_units_centered[j]) > np.absolute(p_new[1] - p[1])) \
                and (np.absolute(x_units_centered[i]) > np.absolute(p_new[0]-p[0])):
                    bool_mask[j][i] = True

@jit
def calculate_intensity(masked_data, x_units_centered, y_units_centered, data):
    intensities, x_total, y_total, total_intensity = 0, 0, 0, 0
    for j in range(len(masked_data)):
        for i in range(len(masked_data[j])):
            point_intensity = data[j][i]
            total_intensity += point_intensity
            y_total += y_units_centered[j]*point_intensity
            x_total += x_units_centered[i]*point_intensity
    return x_total, y_total, total_intensity


def main():
    # d = Data(data_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp253_scan0015_0001.xml",
    #          center_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp253_scan0010_0001.xml",
    #          background_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp253_scan0011_0001.xml")
    # d = Data(data_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/HiResSANS_exp9_scan0030_0001.xml",
    #          center_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/HiResSANS_exp9_scan0006_0001.xml",
    #          background_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/HiResSANS_exp9_scan0038_0001.xml")
    d = Data(data_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp318_scan0229_0001.xml",
             center_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp318_scan0229_0001.xml")
    get_center_of_mass(d)
if __name__ == "__main__":
    main()
