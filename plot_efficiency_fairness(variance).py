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
        search_test_accu = re.search( r'At round '+pattern+' testing variance: '+pattern, line, re.M|re.I)
        if search_test_accu:
            rounds.append(int(search_test_accu.group(1)))
            accu.append(10000*float(search_test_accu.group(2)))
            
        # search_loss = re.search(r'At round '+pattern+' training loss: '+pattern, line, re.M|re.I)
        # if search_loss:
        #     loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: '+pattern, line, re.M|re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))

    return rounds, loss, accu, accu_train



accuracies = [ 
"./log_fmnist/qffedavg_samp1_run1_q0__3",
"./log_fmnist/hgfed_samp1_run1_q0__3",
"./log_fmnist/afl_samp1_run1_q0__3",
"./log_fmnist/qffedavg_samp1_run1_q5__3",
"./log_fmnist/qffedavg_samp1_run1_q15__3",
# "./log_synthetic/qffedavg_samp1_run1_q0__3",
# "./log_synthetic/hgfed_samp1_run1_q0__3",
# "./log_synthetic/afl_samp1_run1_q0__3",
# "./log_synthetic/qffedavg_samp1_run1_q1__3",
# "./log_adult/qffedavg_samp1_run1_q0__3",
# "./log_adult/hgfed_samp1_run1_q0__3",
# "./log_adult/afl_samp1_run1_q0__3",
# "./log_adult/qffedavg_samp1_run1_q0.01__3",
# "./log_adult/qffedavg_samp1_run1_q2__3",
# "./log_vehicle/qffedavg_samp1_run1_q0__3",
# "./log_vehicle/hgfed_samp1_run1_q0__3",
# "./log_vehicle/afl_samp1_run1_q0__3",
# "./log_vehicle/qffedavg_samp1_run1_q5__3"
]

# dataset = ["vehicle"]
# dataset = ["synthetic"]
# dataset = ["Adult"]
dataset = ["FMNist"]


f = plt.figure(figsize=[5.5, 4.5])

sampling_rate=[1]


rounds0, losses0, test_accuracies0, train_accuracies0 = parse_log(accuracies[0])
rounds1, losses1, test_accuracies1, train_accuracies1 = parse_log(accuracies[1])
rounds2, losses2, test_accuracies2, train_accuracies2 = parse_log(accuracies[2])
rounds3, losses3, test_accuracies3, train_accuracies3 = parse_log(accuracies[3])
rounds4, losses4, test_accuracies4, train_accuracies4 = parse_log(accuracies[4])

plt.plot(np.asarray(rounds0), np.asarray(test_accuracies0), linewidth=1.0, label=r'FedAvg', color="#d62728")
plt.plot(np.asarray(rounds3), np.asarray(test_accuracies3), linewidth=1.0, label=r'q-FedAvg, q=5', color="#063970")
plt.plot(np.asarray(rounds4), np.asarray(test_accuracies4), linewidth=1.0, label=r'q-FedAvg, q=15', color="#76b5c5")
plt.plot(np.asarray(rounds1)[::sampling_rate[0]], np.asarray(test_accuracies1)[::sampling_rate[0]],  '--', linewidth=1.0, label=r'lhfed, n=1')
#plt.plot(np.asarray(rounds1), np.asarray(test_accuracies1),  '--', linewidth=3.0, label=r'hgfed, q=0')
plt.plot(np.asarray(rounds2), np.asarray(test_accuracies2),  '.-', linewidth=1.0, label=r'afl')

plt.ylabel('Variance of Test Accuracy', fontsize=15)
plt.xlabel('# Rounds', fontsize=22)

plt.legend(loc='best', frameon=False)
plt.title(dataset[0], fontsize=22, fontweight='bold')

plt.xlim(0, len(rounds0))
#plt.xlim(0, 10)
plt.tight_layout()

#f.savefig("efficiency-variance__lhfed-qffedavg-afl__vehicle.pdf")
# f.savefig("efficiency-variance__lhfed-qffedavg-afl__synthetic.pdf")
# f.savefig("efficiency-variance__lhfed-qffedavg-afl__adult.pdf")
f.savefig("efficiency-variance__lhfed-qffedavg-afl__fmnist.pdf")
