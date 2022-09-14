import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot

matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)


def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    sim = []
    accu_train = []

    test_accu_1 = []
    test_accu_2 = []
    train_accu_1 = []
    train_accu_2 = []

    for line in open(file_name, 'r'):
        pattern = '(.*)'
        search_test_accu = re.search(r'At round ' + pattern + ' testing accuracy: ' + pattern, line, re.M | re.I)
        if search_test_accu:
            rounds.append(int(search_test_accu.group(1)))
            accu.append(float(search_test_accu.group(2)))

        # search_loss = re.search(r'At round '+pattern+' training loss: '+pattern, line, re.M|re.I)
        # if search_loss:
        #     loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: ' + pattern, line, re.M | re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))
    # accu[:] = [accu[len(accu)-1] - e for e in accu]
    return rounds, loss, accu, accu_train


accuracies = [
    # "./log_fmnist/qffedavg_samp1_run1_q0",
    # "./log_fmnist/hgfed_samp1_run1_q0",
    # "./log_fmnist/afl_samp1_run1_q0",
    # "./log_fmnist/qffedavg_samp1_run1_q5",
    # "./log_fmnist/qffedavg_samp1_run1_q15",

    # "./log_synthetic/hgfed_samp1_run1_q0",
    # "./log_synthetic/hgfednon_samp1_run1_q0",

    # "./log_adult/qffedavg_samp1_run1_q0",
    # "./log_adult/hgfed_samp1_run1_q0",
    # "./log_adult/afl_samp1_run1_q0",
    # "./log_adult/qffedavg_samp1_run1_q0.01",
    # "./log_adult/qffedavg_samp1_run1_q2",


    "./log_vehicle/hgfed_samp1_run1_q0",
    "./log_vehicle/hgfednon_samp1_run1_q0",
]

dataset = ["Vehicle"]
# dataset = ["Synthetic"]
# dataset = ["Adult"]
# dataset = ["Fashion MNIST"]

f = plt.figure(figsize=[5.5, 4.5])

sampling_rate = [1]

rounds0, losses0, test_accuracies0, train_accuracies0 = parse_log(accuracies[0])
rounds1, losses1, test_accuracies1, train_accuracies1 = parse_log(accuracies[1])

max0 = max(test_accuracies0)
max1 = max(test_accuracies1)

final_max = max(max0, max1)
test_accuracies0[:] = [final_max - e for e in test_accuracies0]
test_accuracies1[:] = [final_max - e for e in test_accuracies1]

# test_accuracies4[:] = [final_max - e for e in test_accuracies4]

plt.plot(np.asarray(rounds0)[::sampling_rate[0]], np.asarray(test_accuracies0)[::sampling_rate[0]], '--', linewidth=1.0, label=r'lhfed (selection of fittest), n=2')
plt.plot(np.asarray(rounds1)[::sampling_rate[0]], np.asarray(test_accuracies1)[::sampling_rate[0]], '--', linewidth=1.0, label=r'lhfed, n=2', color="#d62728")

plt.ylabel('Distance to Empirical Optimal', fontsize=15)
plt.xlabel('# Rounds', fontsize=22)

plt.legend(loc='best', frameon=False)
plt.title(dataset[0], fontsize=22, fontweight='bold')

plt.xlim(0, len(rounds0))
# plt.xlim(0, 10)
plt.tight_layout()

f.savefig("convergence-speed__lhfed-lhfednon__vehicle.pdf")
# f.savefig("convergence-speed__hgfed-hgfednon__synthetic.pdf")
# f.savefig("convergence-speed__hgfed-qffedavg-afl__adult.pdf")
# f.savefig("convergence-speed__lhfed-qfedavg15-afl__fmnist.pdf")
