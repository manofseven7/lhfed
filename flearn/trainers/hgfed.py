import random

import os
import numpy as np
import time
from tqdm import trange, tqdm
import tensorflow as tf
from matplotlib.pyplot import plot
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fair fed avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        self.file = open(self.output, 'w')
        self.file2 = open(self.output + '__2', 'w')
        self.file3 = open(self.output + '__3', 'w')
        self.file4 = open(self.output + '__4', 'w')
        #self.num_rounds = 50
        lp = 20
        batches = {}
        for c in self.clients:
            batches[c] = gen_epoch(c.train_data, self.num_rounds + 2)

        num_clients = len(self.clients)
        pk = np.ones(num_clients) * 1.0 / num_clients
        av_arr = []
        history = {'loss': {},
                   'grad': {}
                   }
        time1 = time.time()

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0:
                num_test, num_correct_test = self.test()  # have set the latest model for all clients
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
                rtsa = '\nAt round {} testing accuracy: {} and testing accuracy per client: {} and var: {}'.format(i,
                                                                                                                   ts_acc,
                                                                                                                   ts_acc_per_client,
                                                                                                                   np.var(
                                                                                                                       ts_acc_per_client))
                rtsa2 = 'At round {} testing accuracy: {}'.format(i, ts_acc)
                rtsa3 = 'At round {} testing ten percent lowest accuracy: {}'.format(i, np.average(ts_acc_per_client[
                                                                                                       ts_acc_per_client < np.percentile(
                                                                                                           ts_acc_per_client,
                                                                                                           lp)]))  # min(ts_acc_per_client))
                rtsa5 = 'At round {} testing ten percent highest accuracy: {}'.format(i, np.average(ts_acc_per_client[
                                                                                                        ts_acc_per_client > np.percentile(
                                                                                                            ts_acc_per_client,
                                                                                                            lp)]))  # min(ts_acc_per_client))
                rtsa4 = 'At round {} testing variance: {}'.format(i, np.var(ts_acc_per_client))
                tqdm.write(rtsa)
                self.file.write(rtsa2 + os.linesep)
                self.file2.write(rtsa3 + os.linesep)
                self.file3.write(rtsa4 + os.linesep)
                self.file4.write(rtsa5 + os.linesep)
                rtra = 'At round {} training accuracy: {} and testing accuracy per client: {} and var: {}'.format(i,
                                                                                                                  tr_acc,
                                                                                                                  tr_acc_per_client,
                                                                                                                  np.var(
                                                                                                                      tr_acc_per_client))
                rtra2 = 'At round {} training accuracy: {}'.format(i, tr_acc)
                rtra3 = 'At round {} training ten percent lowest accuracy: {}'.format(i, np.average(tr_acc_per_client[
                                                                                                        tr_acc_per_client < np.percentile(
                                                                                                            tr_acc_per_client,
                                                                                                            lp)]))  # min(tr_acc_per_client))
                rtra5 = 'At round {} training ten percent highest accuracy: {}'.format(i, np.average(tr_acc_per_client[
                                                                                                         tr_acc_per_client > np.percentile(
                                                                                                             tr_acc_per_client,
                                                                                                             lp)]))  # min(tr_acc_per_client))
                rtra4 = 'At round {} training variance: {}'.format(i, np.var(tr_acc_per_client))
                tqdm.write(rtra)
                self.file.write(rtra2 + os.linesep)
                self.file2.write(rtra3 + os.linesep)
                self.file3.write(rtra4 + os.linesep)
                self.file4.write(rtra5 + os.linesep)
                rvla = 'At round {} validating accuracy: {} and testing accuracy per client: {} and var: {}'.format(i,
                                                                                                                    valid_acc,
                                                                                                                    valid_acc_per_client,
                                                                                                                    np.var(
                                                                                                                        valid_acc_per_client))
                rvla2 = 'At round {} validating accuracy: {}'.format(i, valid_acc)
                rvla3 = 'At round {} validating ten percent lowest accuracy: {}'.format(i, np.average(
                    valid_acc_per_client[
                        valid_acc_per_client < np.percentile(valid_acc_per_client, lp)]))  # min(valid_acc_per_client))
                rvla5 = 'At round {} validating ten percent highest accuracy: {}'.format(i, np.average(
                    valid_acc_per_client[
                        valid_acc_per_client > np.percentile(valid_acc_per_client, lp)]))  # min(valid_acc_per_client))
                rvla4 = 'At round {} variance: {}'.format(i, np.var(valid_acc_per_client))
                tqdm.write(rvla)
                self.file.write(rvla2 + os.linesep)
                self.file2.write(rvla3 + os.linesep)
                self.file3.write(rvla4 + os.linesep)
                self.file4.write(rvla5 + os.linesep)
                self.statistics(i, ts_acc=ts_acc, tr_acc=tr_acc, valid_acc=valid_acc,
                                ts_acc_per_client=ts_acc_per_client, tr_acc_per_client=tr_acc_per_client,
                                valid_acc_per_client=valid_acc_per_client)

            if i % self.log_interval == 0 and i > int(self.num_rounds / 2):
                test_accuracies = np.divide(np.asarray(num_correct_test), np.asarray(num_test))
                np.savetxt(self.output + "_" + str(i) + "_test.csv", test_accuracies, delimiter=",")
                train_accuracies = np.divide(np.asarray(num_correct_train), np.asarray(num_train))
                np.savetxt(self.output + "_" + str(i) + "_train.csv", train_accuracies, delimiter=",")
                validation_accuracies = np.divide(np.asarray(num_correct_val), np.asarray(num_val))
                np.savetxt(self.output + "_" + str(i) + "_validation.csv", validation_accuracies, delimiter=",")
            #################################################################################################################
            indices, selected_clients = self.select_clients(round=i, pk=pk, num_clients=self.clients_per_round)
            selected_clients = selected_clients.tolist()

            # rand = random.random()
            # if rand>=0 and rand <= 0.9:
            #     selected_clients = random.sample(selected_clients, len(selected_clients) - random.randint(1, 9))
            b1l = []
            b2l = []
            for c in selected_clients:
                # communicate the latest model
                # c.set_params(self.latest_model)
                if i == 0:
                    c.set_params(self.latest_model)
                else:
                    weights_before = c.get_params()
                    loss_before = c.get_loss()
                    c.set_params(self.latest_model)
                    loss_after = c.get_loss()
                    print(c.id, " loss local: ", loss_before, " , ", c.id, "loss global: ", loss_after)
                    if loss_before > loss_after:
                        c.set_params(self.latest_model)
                    else:
                        print("Selection of fittest on client ", c.id)
                        c.set_params(weights_before)
                weights_before = c.get_params()
                loss = c.get_loss()  # compute loss on the whole training data, with respect to the starting point (the global model)
                print("i::::::::::::::::::::::::::::::::::::::::::::::::::::::::", i)
                print("loss[" + c.id + "]::::", loss)
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                new_weights = soln[1]

                # plug in the weight updates into the gradient
                grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]

                # batch = next(batches[c])
                # _, grads, loss = c.solve_sgd(batch)
                # grads = grads[1]
                b1l.append(loss)
                b2l.append(loss ** 2)
                if c.id not in history['loss']:
                    history['loss'][c.id] = []
                    history['loss'][c.id].append(loss)
                else:
                    history['loss'][c.id].append(loss)
                if c.id not in history['grad']:
                    history['grad'][c.id] = []
                    history['grad'][c.id].append(grads)
                else:
                    history['grad'][c.id].append(grads)

            av_clients = {}
            n = 5

            for c in selected_clients:
                av_clients[c.id] = np.divide(np.sum(history['loss'][c.id]), len(history['loss'][c.id]))
                # av_clients[c.id] = np.divide(np.sum(history['loss'][c.id]), len(history['loss'][c.id])) if i < n else \
                #     np.divide(np.sum(history['loss'][c.id][i - n + 1:i + 1]),
                #               len(history['loss'][c.id][i - n + 1:i + 1]))
                # print("i::::::::::::::::::::::::::::::::::::::::::::::::::::::::", i)
                # print("n::::::::::::::::::::::::::::::::::::::::::::::::::::::::", n)
                # print("history['loss'][" + c.id + "]::::", history['loss'][c.id])
                # print("history['loss'][" + c.id + "][i - n:i]::::", history['loss'][c.id][i - n + 1:i + 1])
                # print("Jadid av_clients[" + c.id + "]===", av_clients[c.id])
                # print("Qadim av_clients[" + c.id + "]===", np.divide(np.sum(history['loss'][c.id]), len(history['loss'][c.id])))
                av_arr.append(av_clients[c.id])

            B = sum(b2l)/sum(b1l)
            aggregate = 0
            # av_sum = np.sum(av_arr) / len(av_arr)

            for c in selected_clients:
                if i < n:
                    av_sum = np.sum(av_arr) * 1600
                else:
                    av_sum = (np.sum(av_arr[i - n:i]) / n) * 1600

                grad = history['grad'][c.id][len(history['grad'][c.id]) - 1]

                # grad = np.add(np.subtract(history['grad'][c.id][len(history['grad'][c.id]) - 1],
                #                           history['grad'][c.id][len(history['grad'][c.id]) - 2]),
                #               np.average(history['grad'][c.id], axis=0)) \
                #     if i > 0 else \
                #     history['grad'][c.id][len(history['grad'][c.id]) - 1]

                aggregate += np.multiply(grad, (history['loss'][c.id][len(history['loss'][c.id]) - 1]/B))#np.divide(av_clients[c.id], av_sum))
            print("b:", B)
            self.latest_model = np.subtract(self.latest_model, aggregate)
        time2 = time.time()
        self.do_statistics(time=time2 - time1)
        ###########################################################################################################
