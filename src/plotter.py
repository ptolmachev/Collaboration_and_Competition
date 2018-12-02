
from scipy.signal import savgol_filter
import numpy as np
from matplotlib import pyplot as plt
def plotter(scores, threshold, filt = False):

    #PLOT SCORES
    x = np.arange(len(scores))
    xs = threshold*np.ones(len(scores))
    plt.xkcd()
    plt.plot(xs,'k--', linewidth = 2)

    if filt == True:
        plt.plot(savgol_filter(scores,51,3), linewidth = 3, color = 'Red')
    else:
        plt.plot(scores, linewidth=3, color='Red')
    plt.title("Gathered reward per episode")
    plt.xlabel("Number of an episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()
#
# import pickle
# scores = pickle.load(open('../results/Tennis.pkl', 'rb+'))
# plotter(scores, 0.5, filt = True)