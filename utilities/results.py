import os
import csv
import numpy as np

def print_result(file):
    f = open(file)  # encoding="utf8"
    reader = csv.DictReader(f, delimiter=",")
    data = [row for row in reader]
    print(data)


class Results():

    def __init__(self, args, test_loss, test_accuracy, test_accuracy_ema, adv_test_accuracy, adv_test_accuracy_ema):
        self.args = args
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        self.test_accuracy_ema = test_accuracy_ema
        self.adv_test_accuracy = adv_test_accuracy
        self.adv_test_accuracy_ema = adv_test_accuracy_ema


    def save_results_normal(self, file, seed=0, mu=0, sigma = 0):

        names = [
            "exp",
            "train_mode",
            "ema_mode",
            "adv_mode",
            "seed",
            "lr",
            "epochs",
            "dataset",
            "network",
            "test_loss",
            "test_acc",
            "adv_test_accuracy",
            "test_accuracy_ema",
            "adv_test_accuracy_ema",
            "mu",
            "sigma",
            "model_dir",
        ]

        values = [
            self.args.exp_identifier,
            self.args.train_mode,
            self.args.ema_mode,
            self.args.adv_mode,
            seed,
            self.args.lr,
            self.args.epochs,
            self.args.dataset,
            self.args.model_architecture,
            self.test_loss,
            self.test_accuracy * 100,
            self.adv_test_accuracy * 100,
            self.test_accuracy_ema * 100,
            self.adv_test_accuracy_ema * 100,
            mu,
            sigma,
            os.path.join(self.args.experiment_name, 'checkpoints'),
        ]

        if os.path.isfile(file):
            with open(file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(values)
        else:
            np.savetxt(file, (names, values), delimiter=",", fmt="%s")


    def save_results_adv(args, file, test_loss_m1, test_accuracy_m1, adv_accuracy_m1,
                            test_loss_m2, test_accuracy_m2, seed=0, mu=0, sigma = 0):

        names = [
            "exp",
            "mode",
            "seed",
            "lr",
            "epochs",
            "dataset",
            "network",
            "classifier",
            "test_loss_m1",
            "test_acc_m1",
            "adv_test_acc_m1",
            "test_loss_m2",
            "test_acc_m2",
            "mu",
            "sigma",
            "adv_loss_type",
            "epsilon",
            "model_dir",
        ]

        values = [
            args.exp_identifier,
            args.mode,
            seed,
            args.lr,
            args.epochs,
            args.dataset,
            args.model1_architecture,
            args.ft_prior,
            test_loss_m1,
            test_accuracy_m1 * 100,
            adv_accuracy_m1 * 100,
            test_loss_m2,
            test_accuracy_m2 * 100,
            mu,
            sigma,
            args.adv_loss_type,
            args.epsilon,
            os.path.join(args.experiment_name, 'checkpoints'),
        ]

        if os.path.isfile(file):
            with open(file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(values)
        else:
            np.savetxt(file, (names, values), delimiter=",", fmt="%s")