from Parser import Parser
from Operations import Operations
from Display import Display
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd


class Data(object):
    def __init__(self, data_file, center_file=None, background_file=None):
        self.data_f = data_file
        if center_file is not None:
            self.center_f = center_file
        if background_file is not None:
            self.backgrd_f = background_file

    @staticmethod
    def get_data(p):
        detector_data = np.rot90(np.array(p.xpath_get("/SPICErack/Data/Detector/data")))
        # Uncomment for files with a sample_det_dist and sample_to_flange
        # distance_1 = p.xpath_get("/SPICErack/Motor_Positions/sample_det_dist/#text")
        # distance_2 = p.xpath_get("/SPICErack/Header/sample_to_flange/#text")
        pixel_size_x = p.xpath_get("/SPICErack/Header/x_mm_per_pixel/#text")
        pixel_size_y = p.xpath_get("/SPICErack/Header/y_mm_per_pixel/#text")
        translation = p.xpath_get("/SPICErack/Motor_Positions/detector_trans/#text")
        # detector_wing = p.xpath_get("/SPICErack/Data/DetectorWing/data")
        return (detector_data, pixel_size_x, pixel_size_y, translation)
        # , detector_wing)

    def setup(self):
        """
        sets up the data for the three files
        """
        self.p_data = Parser(self.data_f)
        self.data = xr.DataArray(Data.get_data(self.p_data)[0], dims=["x", "y"])
        # x_axis_units, y_axis_units = Operations.get_axes_units(data_shape=self.detector_wing.shape,
        #                                                         pixel_size=[pixel_size_x, pixel_size_y])
        # self.detector_wing = xr.DataArray(Data.get_data(self.p_data)[4],
        #                                   coords=[y_axis_units, x_axis_units],
        #                                   dims=["y", "x"])
        # self.wing_center = Operations.find_center(self.detector_wing,
        #                                           self.size, self.translation)

        p_center = Parser(self.center_f)
        size_x = Data.get_data(p_center)[1]
        size_y = Data.get_data(p_center)[2]
        self.size = (size_x, size_y)
        self.translation = Data.get_data(p_center)[3]

        self.center_data = Data.get_data(p_center)[0]
        x_axis_units, y_axis_units = Operations.get_axes_units(
                                            data_shape=self.data.values.shape,
                                            pixel_size=[size_x, size_y])

        y_axis_units = y_axis_units + self.translation
        self.center_data = xr.DataArray(
                                    self.center_data,
                                    coords=[x_axis_units, y_axis_units],
                                    dims=['x', 'y'])
        self.center = Operations.find_center(self.center_data, self.size,
                                                 self.translation)
        self.data.x.values = self.center_data.x.values
        self.data.y.values = self.center_data.y.values

            # self.wing_center = Operations.find_center(self.detector_wing,
            #                                           self.size,
            #                                           self.translation)


        try:
            p_backgrd = Parser(self.backgrd_f)
            self.backgrd_data = xr.DataArray(Data.get_data(p_backgrd)[0],
                                             dims=['x', 'y'])
            self.backgrd_data.y.values = self.data.y.values
            self.backgrd_data.x.values = self.data.x.values
            self.subtracted_data = self.data - self.backgrd_data
        except:
            pass

    def display(self):
        """
        Graphs a plotly line graph
        """
        p = Parser(self.data_f)
        profile = Operations.integrate(size=(self.size),
                                       center=(self.center[2], self.center[3]),
                                       data=self.subtracted_data)

        Display.plot1d(com=(self.center[2], self.center[3]),
                       difference=self.subtracted_data.values,
                       profile=profile,
                       pixel_size=(self.size[0], self.size[1]))

    def display2d(self):
        """
        Graphs a plotly heatmap and/or a matplotlib image of
        the subtracted data.
        If there's no subtracted data, will graph beam center.
        """
        pixel_size_x, pixel_size_y = self.size
        try:
            Display.plot2d(data=self.subtracted_data,
                           parameters=(self.size[0],
                                       self.size[1], self.translation),
                           center=(self.center[0], self.center[1]))
        except:
            Display.plot2d(data=self.center_data, parameters=(self.size[0],
                           self.size[1], self.translation),
                           center=self.center)

    def solid_angle(self):
        try:
            correct = Operations.solid_angle_correction(
                        center=self.center,
                        data=self.subtracted_data)
        except:
            correct = Operations.solid_angle_correction(center=self.center,
                                                        data=self.data)
        return correct

    @staticmethod
    def sensitivity(p_flood, p_sample, p_dark):
        flood_data = Data.get_data(p_flood)[0]
        flood_data = np.array(Data.get_data(p_flood)[0])

        masked_cols = np.arange(105, flood_data.shape[1])
        mask = np.zeros_like(flood_data)
        mask[:, masked_cols] = 1

        sample_data = np.array(Data.get_data(p_sample)[0])
        sample_data = np.ma.masked_array(sample_data, mask)

        dark = np.array(Data.get_data(p_dark)[0])
        dark = np.ma.masked_array(dark, mask)

        sensitivity = np.array(Operations.calculate_sensitivity(flood_data))
        sensitivity = np.ma.masked_array(sensitivity, mask)
        sensitivity = np.log(sensitivity)

        new_sample = Operations.correct_for_sensitivity(sample=sample_data,
                                                        flood_data=flood_data,
                                                        dark_current=dark,
                                                        min_sensitivity=0.5,
                                                        max_sensitivity=1.5)

        new_sample = np.log(np.array(new_sample))
        new_sample = np.ma.masked_array(new_sample, mask)
        flood_data = np.ma.masked_array(flood_data, mask)

        fig = plt.figure(figsize=(20, 15))
        ax1 = fig.add_subplot(221)
        im = ax1.imshow(np.log(flood_data))
        fig.colorbar(im)
        ax1.set_title("Flood Data")
        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(np.log(sample_data))
        fig.colorbar(im2)
        ax2.set_title("Sample Data")
        ax3 = fig.add_subplot(223)
        im3 = ax3.imshow(sensitivity)
        fig.colorbar(im3)
        ax3.set_title("Sensitivity")
        ax4 = fig.add_subplot(224)
        im4 = ax4.imshow(new_sample)
        ax4.set_title("Sensitivity Correction")
        fig.colorbar(im4)
        plt.figure()
        plt.plot(new_sample.sum(axis=1))
        plt.figure()
        plt.plot(np.log(sample_data).sum(axis=1))
        plt.show()

        return new_sample


def main():
    # d = Data(data_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp275_scan0001_0001.xml",
    #          center_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp275_scan0000_0001.xml")
    # d = Data(data_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp318_scan0229_0001.xml",
    #          center_file="C:/Users/tsy/Documents/GitHub/Data-Grapher-Xarray/Data Examples/BioSANS_exp318_scan0229_0001.xml")
    d = Data(data_file="C:/Users/tsy/Documents/GitHub/Data_Grapher/Data Examples/HiResSANS_exp9_scan0030_0001.xml",
         center_file="C:/Users/tsy/Documents/GitHub/Data_Grapher/Data Examples/HiResSANS_exp9_scan0006_0001.xml",
        background_file="C:/Users/tsy/Documents/GitHub/Data_Grapher/Data Examples/HiResSANS_exp9_scan0038_0001.xml")
    d.setup()
    # d.solid_angle()
    # d.xarray_plot()
    # d.display()
    d.display2d()

    p_flood = Parser("Data Examples/BioSANS_exp318_scan0008_0001.xml")
    p_sample = Parser("Data Examples/BioSANS_exp318_scan0229_0001.xml")
    p_dark = Parser("Data Examples/BioSANS_exp318_scan0009_0001.xml")
    # Data.sensitivity(p_flood=p_flood,
    #                  p_sample = p_sample,
    #                 p_dark = p_dark)

if __name__ == "__main__":
    main()
