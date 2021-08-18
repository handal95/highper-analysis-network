import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, dataset):
        super(Dashboard).__init__()
        self.dataset = dataset
        self.sequence = dataset.seq_len

        self.fig, self.ax = self.init_figure()
        self.data = None
        self.pred = None
        self.pred0 = None
        self.pred1 = None
        self.pred2 = None
        self.pred3 = None
        self.pred4 = None
        self.pred5 = None
        self.area_upper = None
        self.area_lower = None
        self._time = None
        self.time = dataset.time
        self.scope = 480

    def init_figure(self):
        fig, ax = plt.subplots(figsize=(16, 8), facecolor="lightgray")
        fig.suptitle(self.dataset.title, fontsize=25)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        return fig, ax

    def concat(self, target, x, denormalize=False):
        if denormalize is True:
            x = self.dataset.denormalize(x)

        x = x.detach().numpy()
        if target is None:
            target = x
        else:
            target = np.concatenate((target, x))

        return target

    def visualize(self):
        fig, ax = self.fig, self.ax
        fig.show()

        xtick = np.arange(0, self.scope, 24)
        for i in range(self.scope, self.data.shape[0]):
            ax.clear()
            data = self.data[i : i + self.scope, 0]
            pred = self.pred[i : i + self.scope, 0]
            pred1 = self.pred[i : i + self.scope, 1]
            pred2 = self.pred[i : i + self.scope, 2]
            time = self.time[i : i + self.scope : 24, 0]

            plt.xticks(xtick, time, rotation=45)
            ax.plot(data, "r-", label="Actual Data")
            ax.plot(pred, label="Predict Data")
            ax.plot(pred1, label="Predict Data1")
            ax.plot(pred2, label="Predict Data2")
            ax.legend()

            fig.canvas.draw()
            fig.canvas.flush_events()


    def _visualize(self, time, data, pred):
        try:
            fig, ax = self.fig, self.ax

            data = data[0].detach().numpy().ravel()
            pred = pred[0].detach().numpy().ravel()
            
            def concat(target, x, loc=-1):
                if target is None:
                    target = x
                else:
                    target = np.concatenate((target, x[loc:loc+1]))
                return target
            
            self.data = concat(self.data, data, -2)
            min_scope = max(0, self.data.size - self.scope + 1)
            min_axvln = min(self.scope, self.data.size)
            # if self.area_lower is None:
            #     self.area_lower = data[:]
            #     self.area_upper = data[:]

            # self.area_upper = np.append(self.area_upper, pred[7:].max())
            # self.area_lower = np.append(self.area_lower, pred[7:].min())

            self.pred1 = concat(self.pred1, pred, 3)
            self.pred2 = concat(self.pred2, pred, 5)
            self.pred3 = concat(self.pred3, pred, -2)
            # self.pred4 = concat(self.pred4, pred, 11)
            # self.pred5 = concat(self.pred5, pred, -2)
            self._time = concat(self._time, time, 0)
            ax.clear()
            ax.grid()


            ax.vlines(x=min_axvln, ymin=pred[3:].min(), ymax=pred[3:].max(), linewidth=3)

            ax.plot(self.data[min_scope:], "r-", alpha=0.6, linewidth=2, label="Actual Data")
            ax.plot(self.pred1[min_scope:], alpha=0.4, label="Pred 3")
            ax.plot(self.pred2[min_scope:], alpha=0.4, label="Pred 5")
            ax.plot(self.pred3[min_scope:], alpha=0.4, label="Pred 8")
            # ax.plot(self.pred4[min_scope:], alpha=0.4, label="Pred 11")
            # ax.plot(self.pred5[min_scope:], alpha=0.4, label="Pred")
            ax.legend(loc='upper right')

            xtick = np.arange(0, min_axvln, 24)
            # ax.fill_between(np.arange(10), self.area_lower[min_scope:], self.area_upper[min_scope:], color='lightgray', alpha=0.5)
            
            values = self._time[min_scope::24]
            plt.xticks(xtick, values, rotation=45)

            # ax.relim()
            # ax.autoscale_view()

            fig.show()
            fig.canvas.draw()
            fig.canvas.flush_events()
        except (KeyboardInterrupt, AttributeError):
            fig.close()
            raise KeyboardInterrupt
        
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
