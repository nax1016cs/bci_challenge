import numpy as np
import matplotlib.pyplot as plt


def draw(method):
    
    labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
    
    x = np.arange(len(labels))  # the x label locations
    y = np.arange(0.0,1.1, 0.1)  # the y label locations

    width = 0.2  # the width of the bars
    eeg, shallow, scc = get_data(method)
    
    fig, ax = plt.subplots()
    rects2 = ax.bar(x  - width  , shallow, width, label = 'SHALLOWNET')
    rects1 = ax.bar(x           , eeg,     width, label = 'EEGNET')
    rects3 = ax.bar(x + width   , scc,     width, label = 'SCCNET')

    ax.set_title(method)
    ax.set_yticks(y)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def get_data(method):
    eeg = np.zeros(9)
    shallow = np.zeros(9)
    scc = np.zeros(9)

    for i in range(1,11):
        ind_npz = np.load("test_npz/" + method + "_" + str(i) + ".npz")
        t_shallow = ind_npz["shallow"]
        t_scc = ind_npz["scc"]
        t_eeg = ind_npz["eeg"]
        eeg += t_eeg
        scc += t_scc
        shallow += t_shallow
#     print("=========" + method + "=========")
#     print("eeg: ", list(eeg/10))
#     print("scc: ", list(scc/10))
#     print("shallow: ", list(shallow/10))
    acc = 0
    for i in scc:
        acc += i
    print("avg acc: ", acc/100)
    return list(eeg/10), list(shallow/10), list(scc/10)