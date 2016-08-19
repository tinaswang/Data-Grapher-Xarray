from Parser import Parser
from Operations import Operations
from Display import Display
import numpy as np
import xarray as xr
import pandas as pd
import json


class DataConfig(object):
    """
    This is the version of the Data class that includes support for config
    files and pandas
    """
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
        """
        Uses json library to turn config file into a dictionary and make the
        dictionary values the data
        """
        with open(config, 'r') as f:
            f = f.read()
            config_data = json.loads(f)
        data_dict = {}
        data_list = DataConfig.iterate_dict(config_data, p, data_dict)
        return data_list

    @staticmethod
    def iterate_dict(d, p, data_dict):
        """
        Goes through dictionary and updates values with correct data values
        from Parser.xpath_get
        """
        for key, value in d.items():
            if isinstance(value, dict):
                DataConfig.iterate_dict(value, p, data_dict)
            else:
                data_dict.update({key: p.xpath_get(value)})
        return data_dict

    def setup(self):
        """
        sets up the data for the three files
        takes data the dictionary created from the config files
        will also set up detector wing if present
        """
        p_data = Parser(self.data_f)
        d_values = DataConfig.get_data(p_data, self.config)

        self.data = xr.DataArray(np.rot90(d_values['detector_data']),
                                 dims=['y', 'x'])

        size_x = d_values.get("pixel_size_x")
        size_y = d_values.get("pixel_size_y")
        self.size = size_x, size_y
        try:
            self.detector_wing = xr.DataArray(np.rot90(d_values["detector_wing"]), dims=['y', 'x'])
            x_wing, y_wing = Operations.get_axes_units(
                                    data_shape=self.detector_wing.shape,
                                    pixel_size=[size_y, size_x])

            self.detector_wing.y.values = y_wing
            self.detector_wing.x.values = x_wing

            self.radius = d_values.get("radius")
            self.shift = d_values.get("rotation")
        except:
            pass

        try:
            self.p_center = Parser(self.center_f)
            center_values = DataConfig.get_data(self.p_center, self.config)
            self.center_data = np.rot90(center_values['detector_data'])

            self.translation = center_values.get("translation")
            x_axis_units, y_axis_units = Operations.get_axes_units(
                                            data_shape=self.data.values.shape,
                                            pixel_size=[size_y, size_x])

            y_axis_units = y_axis_units + self.translation
            self.center_data = xr.DataArray(
                                    self.center_data,
                                    coords=[y_axis_units, x_axis_units],
                                    dims=['y', 'x'])
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
            self.backgrd_data = xr.DataArray(backgrd, dims=['y', 'x'])

            self.backgrd_data.y.values = self.data.y.values
            self.backgrd_data.x.values = self.data.x.values
            self.subtracted_data = self.data - self.backgrd_data
        except:
            pass

    def display(self):
        """
        Graphs a plotly line graph of the radial profile
        If there's no subtracted data, it will call an exception
        """
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
            raise NameError("Background data not found")

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
        """
        does solid angle correction
        """
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
        """
        performs sensitivity correction
        """
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

    def __setup_df(self, data, size, z=10, name="Detector1",
                   wing=False, radius=1.13, shift=10):
        """
        Private function that sets up the pandas DataFrame
        Calculates theta differently if detector is a wing detector
        """
        size_x = data.shape[1]
        size_y = data.shape[0]
        dim_x = size[1]
        dim_y = size[0]
        offset_x, offset_y = (self.center[0], self.center[1])

        iv, jv = np.mgrid[0:size_x, 0:size_y]
        iv = iv.ravel()
        jv = jv.ravel()

        xv = data.x.values
        yv = data.y.values
        yv, xv = np.meshgrid(yv, xv)
        xv = (xv.ravel())/1000
        yv = (yv.ravel())/1000

        if wing is False:
            zv = np.full_like(xv, z)
            v = np.stack((xv, yv, zv), axis=-1)
            u = np.stack((np.full_like(xv, offset_x),
                          np.full_like(xv, offset_y), zv), axis=-1)
            theta = np.arccos(np.sum(u*v, axis=1)/(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)))
        else:
            rad_array = np.full_like(xv, radius)
            xv_positive = (iv*dim_x)/1000
            zv = -(rad_array**2 - xv_positive**2)

            tube_step_angle_radians = np.arcsin((dim_y/1000)/radius)
            tube_step_angle_degrees = np.degrees(tube_step_angle_radians)

            # Angles:
            angles = [-tube_step_angle_degrees * x for x in range(160)]
            theta = np.repeat(angles, size_y) - shift
            pixel_positions = np.linspace(-0.54825, 0.54825, 256)

        name = ([name] * (size_x * size_y))
        # Concatenate all of them
        allv = np.column_stack((name, iv, jv, xv, yv, zv, theta))
        df = pd.DataFrame(data=allv,
                          columns=['name', 'i', 'j', 'x', 'y', 'z', 'theta'])
        df.set_index(['name', 'i', 'j'], inplace=True)
        return df

    def make_df(self, add_wing=False, z=10):
        """
        function that uses __setup_df to set up pandas DataFrame
        Has option to add wing data
        """
        df = self.__setup_df(self.data, self.size, z=10)
        if add_wing is True:
            df2 = self.__setup_df(data=self.detector_wing, size=self.size,
                                  name="Detector Wing", wing=True, shift=self.shift)
            df = df.append(df2)
        return df2


def main():
    d = DataConfig(
            config="Data Examples/config.json",
            data_file="Data Examples/BioSANS_exp318_scan0229_0001.xml",
            center_file="Data Examples/BioSANS_exp318_scan0008_0001.xml")
    d.setup()

    df = d.make_df(add_wing=True)
    # d.solid_angle()
    # d.display()
    # d.display2d()

if __name__ == "__main__":
    main()
