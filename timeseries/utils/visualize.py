import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

def visualize(batch_size, real_display, fake_display):
    fig = plt.figure(figsize=(2*batch_size, 2*batch_size))

    fig = _visualize(fig, real_display, batch_size=batch_size, titles="Real")
    fig = _visualize(fig, fake_display, batch_size=batch_size, titles="Fake")
    
    plt.show()

def _visualize(fig, timeseries_batch, feature_idx=0, batch_size=4, titles=None):
    offset = 1 if titles == "Real" else 2
    
    for i, series in enumerate(timeseries_batch.detach()):
        ax = fig.add_subplot(batch_size, 2, 2 * i + offset)
        if titles:
            ax.set_title(titles)
        ax.plot(series[:, feature_idx].numpy())
        ax.set_ylim(62, 82)
        fig.canvas.draw()
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return fig
    
    # def visualize(self, data, ano):
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.ion()

    #     y = data["value"]
    #     x = data["timestamp"]

    #     fulldata_x = x.values.reshape(1, -1)
    #     fulldata_y = y.values.reshape(1, -1)
    #     fullanomal = ano.values.reshape(1, -1)

    #     time_pivot = 240

    #     data_x = fulldata_x[:, :time_pivot]
    #     data_y = fulldata_y[:, :time_pivot]
    #     ano_y = fullanomal[:, :time_pivot]

    #     fig = plt.figure(figsize=(16, 8))
    #     ax = plt.subplot(111, frameon=False)
    #     plt.grid(True)

    #     lines = []
    #     for i in range(fulldata_x.shape[0]):
    #         lw = 1 - 2 * i / 20.0
    #         (line,) = ax.plot(data_x[i], data_y[i], "b-", color="k", lw=lw)
    #         lines.append(line)

    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Data")
    #     ax.text(
    #         0.4,
    #         1.0,
    #         self.dataset,
    #         transform=ax.transAxes,
    #         ha="right",
    #         va="bottom",
    #         color="k",
    #         family="sans-serif",
    #         fontweight="bold",
    #         fontsize=16,
    #     )
    #     ax.set_title("normal", loc="right", color="b", fontsize=16)

    #     def update(*args):
    #         fulldata_x[:, :-1] = fulldata_x[:, 1:]
    #         fulldata_y[0, :-1] = fulldata_y[:, 1:]
    #         fullanomal[:, :-1] = fullanomal[:, 1:]

    #         data_x[:, :] = fulldata_x[:, :time_pivot]
    #         data_y[:, :] = fulldata_y[:, :time_pivot]
    #         ano_y[:, :] = fullanomal[:, :time_pivot]

    #         if data_x[:, -1] == fulldata_x[:, -1]:
    #             plt.clf()
    #             plt.close()

    #         for i in range(len(data_y)):
    #             convert = ano_y[:, -2] - ano_y[:, -1]
    #             if convert > 0:
    #                 ax.set_title("normal", loc="right", color="b", fontsize=16)
    #             elif convert < 0:
    #                 ax.set_title(
    #                     "Anomalies Detected",
    #                     loc="right",
    #                     color="r",
    #                     fontweight="bold",
    #                     fontsize=16,
    #                 )

    #             lines[i].set_xdata(data_x[i])
    #             lines[i].set_ydata(data_y[i])
    #             ax.relim()
    #             ax.autoscale_view()

    #     anim = animation.FuncAnimation(fig, update, interval=1)

    # def normalized_visualize(self, time, value, ano):
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.ion()

    #     y = value
    #     x = time

    #     fulldata_x = x.values.reshape(1, -1)
    #     fulldata_y = y.values.reshape(1, -1)
    #     fullanomal = ano.values.reshape(1, -1)

    #     time_pivot = 240

    #     data_x = fulldata_x[:, :time_pivot]
    #     data_y = fulldata_y[:, :time_pivot]
    #     ano_y = fullanomal[:, :time_pivot]

    #     fig = plt.figure(figsize=(16, 8))
    #     ax = plt.subplot(111, frameon=False)
    #     plt.grid(True)

    #     lines = []
    #     for i in range(fulldata_x.shape[0]):
    #         lw = 1 - 2 * i / 20.0
    #         (line,) = ax.plot(data_x[i], data_y[i], "b-", color="k", lw=lw)
    #         lines.append(line)

    #     plt.ylim([-3, 3])
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Data")
    #     ax.text(
    #         0.4,
    #         1.0,
    #         self.dataset,
    #         transform=ax.transAxes,
    #         ha="right",
    #         va="bottom",
    #         color="k",
    #         family="sans-serif",
    #         fontweight="bold",
    #         fontsize=16,
    #     )
    #     ax.set_title("normal", loc="right", color="b", fontsize=16)

    #     def update(*args):
    #         fulldata_x[:, :-1] = fulldata_x[:, 1:]
    #         fulldata_y[0, :-1] = fulldata_y[:, 1:]
    #         fullanomal[:, :-1] = fullanomal[:, 1:]

    #         data_x[:, :] = fulldata_x[:, :time_pivot]
    #         data_y[:, :] = fulldata_y[:, :time_pivot]
    #         ano_y[:, :] = fullanomal[:, :time_pivot]

    #         if data_x[:, -1] == fulldata_x[:, -1]:
    #             plt.clf()
    #             plt.close()

    #         for i in range(len(data_y)):
    #             convert = ano_y[:, -2] - ano_y[:, -1]
    #             if convert > 0:
    #                 ax.set_title("normal", loc="right", color="b", fontsize=16)
    #             elif convert < 0:
    #                 ax.set_title(
    #                     "Anomalies Detected",
    #                     loc="right",
    #                     color="r",
    #                     fontweight="bold",
    #                     fontsize=16,
    #                 )

    #             lines[i].set_xdata(data_x[i])
    #             lines[i].set_ydata(data_y[i])

    #             ax.relim()
    #             ax.autoscale_view()

    #     anim = animation.FuncAnimation(fig, update, interval=1)
    #     input()
