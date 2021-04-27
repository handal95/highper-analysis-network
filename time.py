import pandas as pd
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from app.utils.file import open_csv


def main():
    df = open_csv("D:\datasets\energy\energy.csv")
    df = df[:720]
    fig = make_subplots(rows=4, cols=1, subplot_titles=['dangjin_f', 'dangin_w'])
    fig.append_trace(go.Line(x=df.time, y=df["dangjin_floating"]), row=1, col=1)
    fig.append_trace(go.Line(x=df.time, y=df["dangjin_warehouse"]), row=2, col=1)
    fig.append_trace(go.Line(x=df.time, y=df["dangjin"]), row=3, col=1)
    fig.append_trace(go.Line(x=df.time, y=df["ulsan"]), row=4, col=1)

    fig.update_layout(height=800, width=1600, title_text="Timeseries")
    fig.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("stop")
