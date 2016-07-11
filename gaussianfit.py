from Parser import Parser
from Operations import Operations
import numpy as np
from Data import Data
import scipy.optimize as opt
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y) = xdata_tuple
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def pad_to_square(a, pad_value=0):
    m = a.reshape((a.shape[0], -1))
    padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
    padded[0:m.shape[0], 0:m.shape[1]] = m
    return padded

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    y = 4*y
    x = 5*x
    r = np.sqrt((x - 4*center[0])**2 + (y - 5*center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

parser = Parser("Data Examples/test_0032_trans_0001.xml")
data = np.rot90(Operations.get_data(parser)[0])
data =  pad_to_square(data)

x = np.linspace(0, 255, 256)
y = np.linspace(0, 255, 256)
x, y = np.meshgrid(x, y)

com = Operations.get_com(data)
# print("Center of mass: {}".format(com))
# amplitude, xo, yo, sigma_x, sigma_y, theta, offset
initial_guess = (300,com[1],com[0],4,4,0,0)
popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data.ravel(), p0 = initial_guess)

# print("Fitting results: {}".format(popt))
# print("Fitting Cov Matrix: {}".format(pcov))
# data_fitted = twoD_Gaussian((x, y), *popt)
#

center_x = popt[1]
center_y = popt[2]

# plt.imshow(data)
# plt.contour(x, y, data_fitted.reshape(256, 256))
# plt.scatter(center_x, center_y, color = "white", s = 50)
# plt.show()

d = Data(data_file="Data Examples/BioSANS_exp253_scan0015_0001.xml",
        center_file="Data Examples/BioSANS_exp253_scan0010_0001.xml",
        background_file="Data Examples/BioSANS_exp253_scan0011_0001.xml")
d.setup()
profile = radial_profile(data=d.subtracted_data, center=(center_x, center_y))
radi = np.linspace(0, 3*profile.shape[0] -1, profile.shape[0])
plt.plot(radi, profile)
plt.show()
