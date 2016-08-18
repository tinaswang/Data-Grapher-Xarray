import numpy as np
import scipy.ndimage as ndimage
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as stats
import xarray as xr
from numba import jit


class Operations(object):

    def __init__(self):
        pass

    @staticmethod
    def solid_angle(center, data, z=5):
        shape = data.values.shape
        cos_theta = np.empty((shape[0], shape[1]))
        for i in range(len(data)):
            for j in range(len(data[i])):
                x = data.coords["x"].values[i]
                y = data.coords["y"].values[j]
                cos_theta[i][j] = (((center[0]**2 + center[1]**2 + z**2) ** (1/2)) /
                                   (x**2 + y**2 + z**2)**(1/2))
        cos_theta = cos_theta**3
        return cos_theta

    @staticmethod
    def solid_angle_correction(center, data, z=5):
        cos_theta_cubed = Operations.solid_angle(center, data, z)
        corrected = xr.DataArray(data.values/cos_theta_cubed, dims=["x", "y"])
        corrected.x.values = data.x.values
        corrected.y.values = data.y.values
        corrected.plot()
        plt.show()
        return corrected

    @staticmethod
    def calculate_sensitivity(flood_data, min_sensitivity=0.5,
                              max_sensitivity=0.5):
        num_pixels = (flood_data.shape[0]*flood_data.shape[1] -
                      np.ma.count_masked(flood_data))
        efficiency = flood_data/((1/num_pixels)*np.sum(flood_data))
        try:
            efficiency = np.nan_to_num(efficiency)
            flood_mask = flood_data.mask
            for i in range(len(flood_data)):
                for j in range(len(flood_data[i])):
                    if efficiency[i][j] < min_sensitivity:
                        flood_mask[i][j] = True
                    elif efficiency[i][j] > max_sensitivity:
                        flood_mask[i][j] = True
            flood_data = np.ma.masked_array(flood_data, flood_mask)
            num_pixels = flood_data.compressed().size
            eff_2 = flood_data/((1/num_pixels)*np.sum(flood_data))
            return np.array(eff_2)
        except:
            return np.array(efficiency)

    @staticmethod
    def correct_for_sensitivity(sample, flood_data, dark_current=None,
                                min_sensitivity=0.5, max_sensitivity=1.5):
        if dark_current is not None:
            sample = sample - dark_current
            flood_data = flood_data - dark_current
        sensitivity = Operations.calculate_sensitivity(flood_data,
                                                       min_sensitivity,
                                                       max_sensitivity)
        new_sample = sample / sensitivity
        return new_sample

    @staticmethod
    def get_com(center_data):
        """derived from http://stackoverflow.com/questions/18435003/ndimages-
        center-of-mass-to-calculate-the-position-of-a-gaussian-peak
        gets a guess for the center of mass
        """
        # hist, bins = np.histogram(center_data.ravel(),
        #                           normed=False,
        #                           bins=49000)
        # threshold = bins[np.cumsum(bins) * (bins[1] - bins[0]) > 30000][0]
        # mnorm2d = np.ma.masked_less(center_data, threshold)
        com = ndimage.measurements.center_of_mass(center_data)
        com = [float(i) for i in com]
        return com

    @staticmethod
    def find_center(center_data, size, translation):
        # finds the actual center of mass via Gaussian fitting
        data = center_data.values
        pixel_size_x, pixel_size_y = size
        x = np.linspace(0, 255, 256)
        y = np.linspace(0, 255, 256)
        x, y = np.meshgrid(x, y)

        data = Operations.pad_to_square(data)
        com = Operations.get_com(data)
        initial_guess = (500, com[1], com[0], 4, 4, 0, 0)
        popt, pcov = opt.curve_fit(Operations.twoD_Gaussian, (x, y),
                                   data.ravel(), p0=initial_guess)
        fit = Operations.twoD_Gaussian((x, y), *popt)
        x_diff = (popt[1] - int(round(popt[1])))*pixel_size_y
        y_diff = (popt[2] - int(round(popt[2]))) * pixel_size_x
        center_x = center_data.coords['y'].values[int(round(popt[1]))] + x_diff
        center_y = center_data.coords['x'].values[int(round(popt[2]))] + y_diff
        return center_x, center_y, popt[1], popt[2]

    @staticmethod
    def integrate(size, center, data):
        """This function does the radial integration.
        Derived from http://stackoverflow.com/questions/21242011/most-efficient
        -way-to-calculate-radial-profile
        """
        # y = np.tile(data.y.values, (256, 1))
        # x = np.tile(np.transpose([data.x.values]), 192)

        y, x = np.indices((data.shape))
        r = np.sqrt((y - center[1])**2 + (x - center[0])**2)
        r = r.astype(np.int)

        tbin = np.bincount(r.ravel(), data.values.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    @staticmethod
    def get_axes_units(data_shape, pixel_size):
        """
        pixel_size in mm
        get default units with center as center of the images
        """
        i_center = data_shape[1]/2
        j_center = data_shape[0]/2
        x_axis_units = (np.arange(data_shape[1])-i_center) * pixel_size[1]
        y_axis_units = (np.arange(data_shape[0])-j_center) * pixel_size[0]
        return x_axis_units, y_axis_units

    @staticmethod
    def twoD_Gaussian(xdata_tuple, amplitude, xo, yo,
                      sigma_x, sigma_y, theta, offset):
        """
        2D Gaussian function
        from http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian
            -function-using-scipy-optimize-curve-fit-valueerror-and-m
        """
        (x, y) = xdata_tuple
        a = ((np.cos(theta)**2)/(2*sigma_x**2) +
             (np.sin(theta)**2)/(2*sigma_y**2))
        b = (-(np.sin(2*theta))/(4*sigma_x**2) +
              (np.sin(2*theta))/(4*sigma_y**2))
        c = ((np.sin(theta)**2)/(2*sigma_x**2) +
             (np.cos(theta)**2)/(2*sigma_y**2))
        g = offset + amplitude*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) +
                                         c*((y-yo)**2)))
        return g.ravel()

    @staticmethod
    def pad_to_square(a, pad_value=0):
        """
        Turns numpy array to square
        from http://stackoverflow.com/questions/10871220/making-a-matrix-square
        -and-padding-it-with-desired-value-in-numpy
        """
        m = a.reshape((a.shape[0], -1))
        padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
        padded[0:m.shape[0], 0:m.shape[1]] = m
        return padded


def main():
    pass

if __name__ == "__main__":
    main()
