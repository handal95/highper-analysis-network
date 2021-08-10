import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, dataset):
        super(Dashboard).__init__()
        self.dataset = dataset
        self.fig, self.ax = self.init_figure()

        self.data = None
        self.pred = None
        self.time = None
        self.scope = 480

    def init_figure(self):
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="lightgray")
        fig.suptitle(self.dataset)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        return fig, ax

    def visualize(self, time, newdata, gz):
        fig, ax = self.fig, self.ax

        data = newdata[0].detach().numpy().ravel()
        pred = gz[0].detach().numpy().ravel()
        seq_len = data.shape[0]
        
        def concat(target, x):
            if target is None:
                target = x
            else:
                target = np.concatenate((target, x[-seq_len:]))
            return target
        
        self.data = concat(self.data, data)
        self.pred = concat(self.pred, pred)
        self.time = concat(self.time, time)
        
        min_scope = max(0, self.data.size - self.scope + 1)
        min_axvln = min(self.scope, self.data.size)
        ax.clear()
        ax.grid()

        ax.axvline(x=min_axvln, ymin=-1.0, ymax=1.0)
        ax.plot(self.data[min_scope:], "r-", label="Actual Data")
        ax.plot(self.pred[min_scope:], "b-", label="Generated Data")
        ax.legend()

        xtick = np.arange(0, min_axvln, 24)
        
        values = self.time[min_scope::24]
        plt.xticks(xtick, values, rotation=45)

        # ax.relim()
        ax.autoscale_view()

        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()


def plt_loss(gen_loss, dis_loss, path, num):
    idx = num * 1000
    gen_loss = np.clip(np.array(gen_loss), a_min=-1500, a_max=1500)
    dis_loss = np.clip(-np.array(dis_loss), a_min=-1500, a_max=1500)
    plt.figure(figsize=(9, 4.5))
    plt.plot(gen_loss[idx : idx + 1001], label="g_loss", alpha=0.7)
    plt.plot(dis_loss[idx : idx + 1001], label="d_loss", alpha=0.7)
    plt.title("Loss")
    plt.legend()
    plt.savefig(
        path + "/Loss_" + str(num) + ".png", bbox_inches="tight", pad_inches=0.5
    )
    plt.close()


def plt_progress(real, fake, epoch, path):
    real = np.squeeze(real)
    fake = np.squeeze(fake)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    fig.suptitle("Data generation, iter:" + str(epoch))
    for i in range(ax.shape[0]):
        ax[i].plot(real[i], color="red", label="Real", alpha=0.7)
        ax[i].plot(fake[i], color="blue", label="Fake", alpha=0.7)
        ax[i].legend()

    plt.savefig(
        path + "/line_generation" + "/Iteration_" + str(epoch) + ".png",
        bbox_inches="tight",
        pad_inches=0.5,
    )
    plt.clf()

