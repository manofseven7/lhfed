import os
import random

import numpy as np
import time

from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fair fed avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        self.file = open(self.output, 'w')
        self.file2 = open(self.output+'__2', 'w')
        self.file3 = open(self.output+'__3', 'w')
        self.file4 = open(self.output+'__4', 'w')
        print('Training with {} workers ---'.format(self.clients_per_round))
        #self.num_rounds = 50
        lp = 20
        num_clients = len(self.clients)
        pk = np.ones(num_clients) * 1.0 / num_clients
        time1 = time.time()

        for i in range(self.num_rounds+1):
            if i % self.eval_every == 0:
                num_test, num_correct_test = self.test() # have set the latest model for all clients
                num_train, num_correct_train = self.train_error()  
                num_val, num_correct_val = self.validate()
                # avg acc
                ts_acc = np.sum(np.array(num_correct_test)) * 1.0 / np.sum(np.array(num_test))
                tr_acc = np.sum(np.array(num_correct_train)) * 1.0 / np.sum(np.array(num_train))
                valid_acc = np.sum(np.array(num_correct_val)) * 1.0 / np.sum(np.array(num_val))
                # per clinet acc
                ts_acc_per_client = np.divide(np.asarray(num_correct_test), np.asarray(num_test))
                tr_acc_per_client = np.divide(np.asarray(num_correct_train), np.asarray(num_train))
                valid_acc_per_client = np.divide(np.asarray(num_correct_val), np.asarray(num_val))
                rtsa = '\nAt round {} testing accuracy: {} and testing accuracy per client: {} and var: {}'.format(i, ts_acc, ts_acc_per_client, np.var(ts_acc_per_client))
                rtsa2 = 'At round {} testing accuracy: {}'.format(i, ts_acc)
                rtsa3 = 'At round {} testing ten percent lowest accuracy: {}'.format(i, np.average(ts_acc_per_client[ts_acc_per_client < np.percentile(ts_acc_per_client, lp)]))#min(ts_acc_per_client))
                rtsa5 = 'At round {} testing ten percent highest accuracy: {}'.format(i, np.average(ts_acc_per_client[ts_acc_per_client > np.percentile(ts_acc_per_client, lp)]))#min(ts_acc_per_client))
                rtsa4 = 'At round {} testing variance: {}'.format(i, np.var(ts_acc_per_client))
                tqdm.write(rtsa)
                self.file.write(rtsa2 + os.linesep)
                self.file2.write(rtsa3 + os.linesep)
                self.file3.write(rtsa4 + os.linesep)
                self.file4.write(rtsa5 + os.linesep)
                rtra = 'At round {} training accuracy: {} and testing accuracy per client: {}'.format(i, tr_acc, tr_acc_per_client, np.var(tr_acc_per_client))
                rtra2 = 'At round {} training accuracy: {}'.format(i, tr_acc)
                rtra3 = 'At round {} training ten percent lowest accuracy: {}'.format(i, np.average(tr_acc_per_client[tr_acc_per_client < np.percentile(tr_acc_per_client, lp)]))#min(tr_acc_per_client))
                rtra5 = 'At round {} training ten percent highest accuracy: {}'.format(i, np.average(tr_acc_per_client[tr_acc_per_client > np.percentile(tr_acc_per_client, lp)]))#min(tr_acc_per_client))
                rtra4 = 'At round {} training variance: {}'.format(i, np.var(tr_acc_per_client))
                tqdm.write(rtra)
                self.file.write(rtra2 + os.linesep)
                self.file2.write(rtra3 + os.linesep)
                self.file3.write(rtra4 + os.linesep)
                self.file4.write(rtra5 + os.linesep)
                rvla = 'At round {} validating accuracy: {} and testing accuracy per client: {}'.format(i, valid_acc, valid_acc_per_client, np.var(valid_acc_per_client))
                rvla2 = 'At round {} validating accuracy: {}'.format(i, valid_acc)
                rvla3 = 'At round {} validating ten percent lowest accuracy: {}'.format(i, np.average(valid_acc_per_client[valid_acc_per_client < np.percentile(valid_acc_per_client, lp)]))#min(valid_acc_per_client))
                rvla5 = 'At round {} validating ten percent highest accuracy: {}'.format(i, np.average(valid_acc_per_client[valid_acc_per_client > np.percentile(valid_acc_per_client, lp)]))#min(valid_acc_per_client))
                rvla4 = 'At round {} validating variance: {}'.format(i, np.var(valid_acc_per_client))
                tqdm.write(rvla)
                self.file.write(rvla2 + os.linesep)
                self.file2.write(rvla3 + os.linesep)
                self.file3.write(rvla4 + os.linesep)
                self.file4.write(rvla5 + os.linesep)
                self.statistics(i, ts_acc=ts_acc, tr_acc=tr_acc, valid_acc=valid_acc,
                                ts_acc_per_client=ts_acc_per_client, tr_acc_per_client=tr_acc_per_client,
                                valid_acc_per_client=valid_acc_per_client)
            if i % self.log_interval == 0 and i > int(self.num_rounds/2):                
                test_accuracies = np.divide(np.asarray(num_correct_test), np.asarray(num_test))
                np.savetxt(self.output + "_" + str(i) + "_test.csv", test_accuracies, delimiter=",")
                train_accuracies = np.divide(np.asarray(num_correct_train), np.asarray(num_train))
                np.savetxt(self.output + "_" + str(i) + "_train.csv", train_accuracies, delimiter=",")
                validation_accuracies = np.divide(np.asarray(num_correct_val), np.asarray(num_val))
                np.savetxt(self.output + "_" + str(i) + "_validation.csv", validation_accuracies, delimiter=",")
            
            
            indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)

            Deltas = []
            hs = []
            selected_clients = selected_clients.tolist()

            # rand = random.random()
            # if rand >= 0 and rand <= 0.1:
            #     selected_clients = random.sample(selected_clients, len(selected_clients) - random.randint(1, 9))

            for c in selected_clients:
                # communicate the latest model
                c.set_params(self.latest_model)
                weights_before = c.get_params()
                loss = c.get_loss()  # compute loss on the whole training data, with respect to the starting point (the global model)
                print("i::::::::::::::::::::::::::::::::::::::::::::::::::::::::", i)
                print("loss[" + c.id + "]::::", loss)
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                new_weights = soln[1]

                # plug in the weight updates into the gradient
                grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]

                Deltas.append([np.float_power(loss + 1e-10, self.q) * grad for grad in grads])

                # estimation of the local Lipchitz constant
                hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * norm_grad(grads) + (
                    1.0 / self.learning_rate) * np.float_power(loss + 1e-10, self.q))


            # aggregate using the dynamic step-size
            self.latest_model = self.aggregate2(weights_before, Deltas, hs)
        time2 = time.time()
        self.do_statistics(time=time2 - time1)

                    



