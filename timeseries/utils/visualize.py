import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self):
        super(Dashboard).__init__()
        self.fig, self.ax = self.init_figure()
        self.data = None
        self.pred = None
        self.scope = 480

    def init_figure(self):
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="lightgray")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Title")

        return fig, ax

    def visualize(self, newdata, gz):
        fig, ax = self.fig, self.ax

        data = newdata[0].detach().numpy()
        pred = gz[0].detach().numpy()

        if self.data is None:
            self.data = data
        else:
            self.data = np.concatenate((self.data, data[-1:]))

        if self.pred is None:
            self.pred = pred
        else:
            self.pred = np.concatenate((self.pred, pred[-1:]))

        min_scope = max(0, self.data.size - self.scope + 1)
        min_axvln = min(self.scope, self.data.size)
        ax.clear()
        ax.grid()

        ax.axvline(x=min_axvln, ymin=-1.0, ymax=1.0)
        ax.plot(self.data[min_scope:], "r-")
        ax.plot(self.pred[min_scope:], "b-")

        ax.relim()
        ax.autoscale_view()

        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()


def visualize(
    batch_size, real_display, fake_display=None, fake_display2=None, block=True
):
    fig = plt.figure(figsize=(16, 8))

    fig = _visualize(fig, real_display, batch_size=batch_size, titles="Real")
    # fig = _visualize(fig, fake_display, batch_size=batch_size, titles="Fake")
    # fig = _visualize(fig, fake_display2, batch_size=batch_size, titles="Fake2")

    plt.show(block=block)
    plt.pause(1)


def _visualize(fig, timeseries_batch, batch_size=4, titles=None):
    offset = 1 if titles == "Real" else 2
    offset = offset + 1 if titles == "Fake2" else offset

    for i, series in enumerate(timeseries_batch.detach()):
        ax = fig.add_subplot(3, 1, offset)
        if titles:
            ax.set_title(titles)
        ax.plot(series.numpy())
        fig.canvas.draw()
        break

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
