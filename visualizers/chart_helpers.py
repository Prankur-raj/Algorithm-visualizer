import plotly.graph_objects as go

from utils.constants import THEME_COLOR, BAR_COLOR, HIGHLIGHT_COLOR


def get_default_layout(title=""):
    return go.Layout(
        title =title,
        plot_bgcolor ="rgba(0,0,0,0)" ,
        paper_bgcolor= "rgba(0,0,0,0)" ,
        font = dict(color=THEME_COLOR),
        showlegend = False
    )


def get_bar_colors(size, highlight_indices=[]):
    colors=[BAR_COLOR]*size
    for i in highlight_indices:
        if 0 <= i < size:   
            colors[i] = HIGHLIGHT_COLOR

   
    return colors

