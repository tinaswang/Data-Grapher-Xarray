from DataConfig import DataConfig
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import json

d = DataConfig(
         config="Data Examples/config.json",
         data_file="Data Examples/BioSANS_exp275_scan0001_0001.xml",
         center_file="Data Examples/BioSANS_exp275_scan0000_0001.xml")
