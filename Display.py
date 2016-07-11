import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import xarray as xr

class Display(object):

    def __init__(self):
        pass

    @staticmethod
    def plot2d(parameters, data, center):
        # makes a 2d plotly graph of the data
        # Will not graph the correct center if data is rotated 90 degrees
        # because of repositioned axes
        pixel_size_x, pixel_size_y, translation = parameters
        x_units_centered=data.coords['x'].values
        y_units_centered = data.coords['y'].values
        X = x_units_centered
        Y =  y_units_centered + translation
        graph_data = [go.Contour(z=np.array(data.values), x=X, y=Y)]
        layout = go.Layout(
         xaxis=dict(title="Milimeters"),
         yaxis=dict(title="Milimeters"),
         showlegend=False,
         annotations=[
                 dict(
                     x=center[0]/pixel_size_x - data.shape[1]/2,
                     y=(center[1])/pixel_size_y,
                     xref='x',
                     yref='y',
                     text="Center",
                     showarrow=True,
                     font=dict(
                         family='Courier New, monospace',
                         size=11,
                         color='#000000'
                     ),
                     align='center',
                     arrowhead=2,
                     arrowsize=1,
                     arrowwidth=2,
                     arrowcolor='#ffffff',
                     ax=20,
                     ay=-30,
                     bordercolor='#c7c7c7',
                     borderwidth=2,
                     borderpad=4,
                     bgcolor='#ffffff',
                     opacity=0.8
                 )
             ]
         )


        fig = go.Figure(data=graph_data, layout=layout)
        py.plot(fig)
        # # Below: matplotlib version
        # plt.imshow(Z)
        # plt.scatter(center[0]/pixel_size_y, center[1]/pixel_size_x,color = "white", s = 50)
        # plt.show()

    @staticmethod
    def plot1d(com, difference, profile, pixel_size):
        # Makes the plotly line graph of the radial integration
        pixel_size_x, pixel_size_y = pixel_size
        length = np.linspace(0, (pixel_size_x*profile.shape[0])/10.0**4.0, profile.shape[0])
        trace = go.Scatter(
            x=length,
            y=profile,
            mode='lines'
            )

        layout = go.Layout(
            xaxis=dict(title="Angle"),
            yaxis=dict(title="Intensity")

        )
        graph_data = [trace]
        fig = go.Figure(data=graph_data, layout=layout)
        py.plot(fig)


def main():
    pass
if __name__ == "__main__":
    main()
