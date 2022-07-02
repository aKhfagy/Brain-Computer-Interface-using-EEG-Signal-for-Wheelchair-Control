from pylab import plot, show, title, legend
from scipy.signal import savgol_filter


def smooth(x, label, window_len=5, show_plot=False):

    y = savgol_filter(x, window_length=window_len, polyorder=2)

    if show_plot:
        title('Signal before and after smoothing for label: ' + str(label))
        plot(x)
        plot(y)
        legend(["Original Signal", "Smoothed Signal"])
        show()
    return y

