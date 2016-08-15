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
        pixel_size_x, pixel_size_y, translation = parameters
        x_units_centered = data.coords['x'].values
        y_units_centered = data.coords['y'].values
        X = y_units_centered
        Y = x_units_centered
        graph_data = [go.Heatmap(z=np.array(data.values), x=X, y=Y)]
        # Contour Graph:
        # graph_data = [go.Contour(z=np.array(data.values),
        # x=X, y=Y, line=dict(smoothing=0))]
        layout = go.Layout(
         xaxis=dict(title="Milimeters"),
         yaxis=dict(title="Milimeters"),
         showlegend=False,
         annotations=[
                 dict(
                     x=center[0],
                     y=center[1],
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

        # Below: matplotlib version
        # plt.imshow(data.values, interpolation = 'none', origin = "lower",
        #            extent=[X[0], X[-1], Y[0], Y[-1]])
        # plt.scatter(center[0], center[1], color = "white", s = 100)
        # plt.set_xlabel("Milimeters")
        # plt.set_ylabel("Milimeters")
        # plt.show()

    @staticmethod
    def plot1d(com, difference, profile, pixel_size):
        # Makes a plotly line graph of the radial integration
        pixel_size_x, pixel_size_y = pixel_size
        length = np.linspace(0, (pixel_size_x*profile.shape[0])/10.0**4.0,
                             profile.shape[0])
        trace = go.Scatter(
            x=(length),
            y=(profile),
            mode='lines'
            )

        layout = go.Layout(
            xaxis=dict(title="Angle"),
            yaxis=dict(title="Intensity")

        )
        graph_data = [trace]
        fig = go.Figure(data=graph_data, layout=layout)
        py.plot(fig)

        # Below: matplotlib version
        # plt.plot(length, profile)
        # plt.set_xlabel("Angle")
        # plt.set_ylabel("Intensity")
        # plt.show()


def main():
    pass
if __name__ == "__main__":
    main()
