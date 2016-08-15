import numpy as np
import pandas as pd
from Data import Data


class DataFrame(object):

    def __init__(self):
        pass

    def setup_df(self, data, size, offset, z=10, name="Detector"):
        size_x = data.shape[1]
        size_y = data.shape[0]
        dim_x = size[0]
        dim_y = size[1]
        offset_x, offset_y = offset

        iv, jv = np.mgrid[0:size_x, 0:size_y]
        iv = iv.ravel()
        jv = jv.ravel()

        xv = data.x.values
        yv = data.y.values
        xv, yv = np.meshgrid(xv, yv)
        xv = xv.ravel()
        yv = yv.ravel()

        zv = np.full_like(xv, z)
        cos_theta = (offset_x**2 + offset_y**2 + z**2)/(xv**2 + yv**2 + zv**2)
        theta = np.nan_to_num(np.arccos(cos_theta))
        print(theta[1000])

        name = ([name] * (size_x * size_y))

        # Concatenate all of them
        allv = np.column_stack((name, iv, jv, xv, yv, zv, theta))
        df = pd.DataFrame(data=allv,
                          columns=['name', 'i', 'j', 'x', 'y', 'z', 'theta'])
        df.set_index(['name', 'i', 'j'], inplace=True)
        return df

    def make_df(self, data_obj, offset, add_wing=False, z=10):
        df = self.setup_df(data_obj.data, data_obj.size, offset, z=10)
        if add_wing is True:
            w_center = (data_obj.wing_center[0], data_obj.wing_center[1])
            df2 = self.setup_df(self, data_obj.detector_wing,
                                data_obj.size, w_center, z,
                                name="Detector Wing")
            df.append(df2, ignore_index=True)
        return df


def main():
    data = Data(data_file="Data Examples/HiResSANS_exp9_scan0030_0001.xml",
                center_file="Data Examples/HiResSANS_exp9_scan0006_0001.xml",
                background_file="Data Examples/HiResSANS_exp9_scan0038_0001.xml")
    data.setup()
    dataframe = DataFrame()
    (dataframe.make_df(data, (data.center[0], data.center[1])))

if __name__ == "__main__":
    main()
