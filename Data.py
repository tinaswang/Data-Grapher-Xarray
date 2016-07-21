from Parser import Parser
from Operations import Operations
from Display import Display
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import xarray as xr
from scipy import ndimage
import time

class Data(object):
    def __init__(self, data_file, center_file=None, background_file=None):
        self.data_f = data_file
        if center_file is not None:
            self.center_f = center_file
        if background_file is not None:
            self.backgrd_f = background_file

    @staticmethod
    def get_data(p):
        detector_data = np.array(p.xpath_get("/SPICErack/Data/Detector/data"))
        # Uncomment for files with a sample_det_dist and sample_to_flange
        # distance_1 = p.xpath_get("/SPICErack/Motor_Positions/sample_det_dist/#text")
        # distance_2 = p.xpath_get("/SPICErack/Header/sample_to_flange/#text")
        pixel_size_x = p.xpath_get("/SPICErack/Header/x_mm_per_pixel/#text")
        pixel_size_y = p.xpath_get("/SPICErack/Header/y_mm_per_pixel/#text")
        translation = p.xpath_get("/SPICErack/Motor_Positions/detector_trans/#text")
        x_axis_units, y_axis_units = Operations.get_axes_units(data_shape=detector_data.shape,
                                                               pixel_size=[pixel_size_y, pixel_size_x])
        y_axis_units = y_axis_units + translation
        detector_data = xr.DataArray(detector_data, coords=[y_axis_units, x_axis_units],
                                     dims=['y', 'x'])
        return (detector_data, pixel_size_x, pixel_size_y,
                translation)

    def setup(self):  # sets up the data for the three files
        p_data = Parser(self.data_f)
        self.data = Data.get_data(p_data)[0]
        # self.data = np.rot90(Data.get_data(self.p_data)[0])
        # for data that needs to be rotated 90 degrees
        try:
            p_center = Parser(self.center_f)
            pixel_size_x = Data.get_data(p_center)[1]
            pixel_size_y = Data.get_data(p_center)[2]
            self.size = (pixel_size_x, pixel_size_y)
            self.translation = Data.get_data(p_center)[3]
            # self.center_data = np.rot90(Data.get_data(p_center)[0])
            # for data that needs to be rotated 90 degrees
            self.center_data = Data.get_data(p_center)[0]
            self.center = Operations.find_center(center_data=self.center_data,
                                                 size=self.size,
                                                 translation=self.translation)
        except:
            pass
        try:
            p_backgrd = Parser(self.backgrd_f)
            self.backgrd_data = Data.get_data(p_backgrd)[0]
            # self.backgrd_data = np.rot90(Data.get_data(p_backgrd)[0])
            # data that needs to be rotated
            self.subtracted_data = self.data - self.backgrd_data
        except:
            pass

    def display(self):  # Graphs a plotly line graph
        try:
            p = Parser(self.data_f)
            profile = Operations.integrate(size=(self.size),
                                           center=(self.center[2], self.center[3]),
                                           data=self.subtracted_data)
            Display.plot1d(com=(self.center[2], self.center[3]),
                           difference=self.subtracted_data.values,
                           profile=profile,
                           pixel_size=(self.size[0], self.size[1]))
        except:
            raise ValueError("Not enough data.")

    def display2d(self):
        # Graphs a plotly heatmap and/or a matplotlib image of the subtracted data
        # If there's no subtracted data, will graph center file
        pixel_size_x, pixel_size_y = self.size
        try:
            Display.plot2d(data=self.subtracted_data,
                           parameters=(self.size[0],
                                       self.size[1], self.translation),
                           center=(self.center[0], self.center[1]))
        except:
            Display.plot2d(data=self.center_data, parameters=(self.size[0],
                            self.size[1], self.translation),
                           center=(self.center[0], self.center[1]))

    @staticmethod
    def sensitivity(flood_field_file, sample, dark_curr_file):
        p_flood = Parser(flood_field_file)
        flood_data = np.rot90(Data.get_data(p_flood)[0].values)
        flood_data = np.log(flood_data)

        p_sample = Parser(sample)
        sample  = np.rot90(Data.get_data(p_sample)[0].values)

        p_dark = Parser(dark_curr_file)
        dark_current = np.rot90(Data.get_data(p_dark)[0].values)

        sensitivity = np.array(Operations.calculate_sensitivity(flood_data, min_sensitivity=0.5, max_sensitivity=1.5))
        sensitivity = np.log(sensitivity)
        new_sample = Operations.correct_for_sensitivity(sample=sample,flood_data=flood_data, dark_current=dark_current, min_sensitivity=0.5,max_sensitivity=1.5)
        new_sample = np.log(np.array(new_sample))
        fig = plt.figure(figsize = (20, 15))
        ax1 = fig.add_subplot(221)
        im = ax1.imshow(flood_data)
        fig.colorbar(im)
        ax1 = fig.add_subplot(222)
        im4 = ax1.imshow(np.log(sample))
        fig.colorbar(im4)
        ax2 = fig.add_subplot(223)
        im2 = ax2.imshow(sensitivity)
        fig.colorbar(im2)
        ax3 = fig.add_subplot(224)
        im3 = ax3.imshow(new_sample)
        fig.colorbar(im3)
        plt.show()

        return new_sample

def main():
    d2 = Data(data_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp275_scan0001_0001.xml",
             center_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp275_scan0000_0001.xml")
    # d = Data(data_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp318_scan0229_0001.xml",
    #          center_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp318_scan0185_0001.xml")
    d = Data(data_file="C:/Users/tsy/Documents/GitHub/Reducer/Data Examples/HiResSANS_exp9_scan0030_0001.xml",
         center_file="C:/Users/tsy/Documents/GitHub/Reducer/Data Examples/HiResSANS_exp9_scan0006_0001.xml",
        background_file="C:/Users/tsy/Documents/GitHub/Reducer/Data Examples/HiResSANS_exp9_scan0038_0001.xml")
    d.setup()
    # d.display()
    # d.display2d()

    result = Data.sensitivity(flood_field_file="Data Examples/BioSANS_exp318_scan0034_0001.xml",
                     sample = "Data Examples/BioSANS_exp318_scan0229_0001.xml",
                     dark_curr_file = "Data Examples/BioSANS_exp318_scan0009_0001.xml")

if __name__ == "__main__":
    main()
