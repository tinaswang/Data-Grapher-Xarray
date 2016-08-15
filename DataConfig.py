from Parser import Parser
from Operations import Operations
from Display import Display
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import json


class DataConfig(object):
    def __init__(self, data_file, center_file=None,
                 background_file=None, config=None):
        self.data_f = data_file
        if center_file is not None:
            self.center_f = center_file
        if background_file is not None:
            self.backgrd_f = background_file
        if config is not None:
            self.config = config

    @staticmethod
    def get_data(p, config):
        with open(config, 'r') as f:
            f = f.read()
            config_data = json.loads(f)
        data_dict = {}
        data_list = DataConfig.iterate_dict(config_data, p, data_dict)
        return data_list

    # for key,val in d.items():
    #    exec(key + '=val')
    # turn key/value into variables
    @staticmethod
    def iterate_dict(d, p, data_dict):
        for key, value in d.items():
            if isinstance(value, dict):
                DataConfig.iterate_dict(value, p, data_dict)
            else:
                data_dict.update({key: p.xpath_get(value)})
        return data_dict

    def setup(self):
        """
        sets up the data for the three files
        """
        p_data = Parser(self.data_f)
        d_values = DataConfig.get_data(p_data, self.config)

        self.data = xr.DataArray(np.rot90(d_values['detector_data']),
                                 dims=['x', 'y'])

        size_x = d_values.get("pixel_size_x")
        size_y = d_values.get("pixel_size_y")
        self.size = size_x, size_y
        try:
            self.detector_wing = xr.DataArray(np.rot90(d_values["detector_wing"]), dims=['x', 'y'])
        except:
            pass

        try:
            self.p_center = Parser(self.center_f)
            center_values = DataConfig.get_data(self.p_center, self.config)
            self.center_data = np.rot90(center_values['detector_data'])

            self.translation = center_values.get("translation")
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

        except:
            pass

        try:
            self.p_backgrd = Parser(self.backgrd_f)
            b_values = DataConfig.get_data(self.p_backgrd, self.config)
            backgrd = np.rot90(b_values['detector_data'])
            self.backgrd_data = xr.DataArray(backgrd, dims=['x', 'y'])

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

        return new_sample

    def __setup_df(self, data, size, z=10, name="Detector"):
        size_x = data.shape[1]
        size_y = data.shape[0]
        dim_x = size[0]
        dim_y = size[1]
        offset_x, offset_y = (self.center[0], self.center[1])

        iv, jv = np.mgrid[0:size_x, 0:size_y]
        iv = iv.ravel()
        jv = jv.ravel()

        xv = data.x.values
        yv = data.y.values
        xv, yv = np.meshgrid(xv, yv)
        xv = xv.ravel()
        yv = yv.ravel()

        zv = np.full_like(xv, z)
        cos_theta = (((offset_x**2 + offset_y**2 + z**2)**(1/2)) /
                     ((xv**2 + yv**2 + zv**2)**(1/2)))
        theta = np.arccos(cos_theta)

        name = ([name] * (size_x * size_y))

        # Concatenate all of them
        allv = np.column_stack((name, iv, jv, yv, xv, zv, theta))
        df = pd.DataFrame(data=allv,
                          columns=['name', 'i', 'j', 'x', 'y', 'z', 'theta'])
        df.set_index(['name', 'i', 'j'], inplace=True)
        return df

    def make_df(self, add_wing=False, z=10):
        df = self.__setup_df(self.data, self.size, z=10)
        if add_wing is True:
            df2 = self.__setup_df(self.detector_wing, self.size,
                                  z, name="Detector Wing")
            df.append(df2, ignore_index=True)
        return df


def main():
    d = DataConfig(
             config="Data Examples/config.json",
             data_file="Data Examples/BioSANS_exp275_scan0001_0001.xml",
             center_file="Data Examples/BioSANS_exp275_scan0000_0001.xml")
    d.setup()
    df = d.make_df()
    print(df)
    # d.solid_angle()
    # d.display()
    # d.display2d()

if __name__ == "__main__":
    main()
