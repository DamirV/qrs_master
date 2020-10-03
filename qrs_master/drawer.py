import numpy as np
import plotly.graph_objs as go


def draw(y):
    x = np.arange(0, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.show()
