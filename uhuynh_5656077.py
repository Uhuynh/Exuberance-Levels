import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import metrics

"""
Author: Uyen Huynh, Student ID: 5656077
Seminar: Financial Instruments: Stocks, Bonds, Derivatives, and Hedge Funds
Goethe University Frankfurt
Submit to Prof. Dr. Jan Viebig
Summer Semester 2020
"""


class SVM:
    """
    Evaluate SVM models on MSCI World Index 'main data' and 'extended data'
    """

    def __init__(self):

        # define global variables for the class
        self.original_data = Helper.get_data()
        self.main_1m, self.main_12m, self.main_60m, self.extended_1m, self.extended_12m, self.extended_60m = \
            Helper.transform(self.original_data)

    def svm_cross_validate(self, horizon='1m', k_fold=10):
        """
        This function performs SVM 10-fold cross-validation on 'main data' and 'extended data'
        of relevant investment horizon
        :param horizon: string (stands for investment horizon we want to perform: 60m, 12m, or 1m)
        :param k_fold: integer (default = 10)
        :return a dataframe consists of accuracy and precision score of SVM with linear kernel & rbf kernel (1.4)
        on 'main data' as well as 'extended data' of the relevant investment horizon
        """

        # get input and output numpy arrays for 'main data' & 'extended data' of relevant investment horizon
        if horizon == '60m':
            x_scaled_main, y_main = Helper.preprocessing(self.main_60m)
            x_scaled_extended, y_extended = Helper.preprocessing(self.extended_60m)
        elif horizon == '12m':
            x_scaled_main, y_main = Helper.preprocessing(self.main_12m)
            x_scaled_extended, y_extended = Helper.preprocessing(self.extended_12m)
        else:
            x_scaled_main, y_main = Helper.preprocessing(self.main_1m)
            x_scaled_extended, y_extended = Helper.preprocessing(self.extended_1m)

        # create an empty dataframe to store the results
        svm_cv = pd.DataFrame(columns=['data', 'investment_horizon', 'kernel', 'accuracy', 'precision'])
        scoring = ['accuracy', 'precision']

        # define a list of classifiers
        # random_state is optional (for re-production purpose)
        dict_classifiers = {
            'linear': SVC(kernel='linear', C=1, random_state=0),
            'rbf_1.4': SVC(kernel='rbf', gamma=1.4, C=1, random_state=0)
        }

        # iterate over 2 classifiers for 'main data'
        for kernel, model_instance in dict_classifiers.items():
            # cross-validation on 'main data'
            # source: https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
            scores = cross_validate(estimator=model_instance,
                                    X=x_scaled_main,
                                    y=y_main,
                                    scoring=scoring,
                                    cv=k_fold,
                                    return_train_score=True)
            svm_cv = svm_cv.append([{'data': 'main data',
                                     'investment_horizon': horizon,
                                     'kernel': kernel,
                                     'accuracy': scores['test_accuracy'].mean(),
                                     'precision': (scores['test_precision'].mean() - 0.75) / 0.25
                                     }], ignore_index=True)

        # iterate over 2 classifiers for 'extended data'
        for kernel, model_instance in dict_classifiers.items():
            # cross-validation on 'extended data'
            scores = cross_validate(estimator=model_instance,
                                    X=x_scaled_extended,
                                    y=y_extended,
                                    scoring=scoring,
                                    cv=k_fold,
                                    return_train_score=True)
            svm_cv = svm_cv.append([{'data': 'extended data',
                                     'investment_horizon': horizon,
                                     'kernel': kernel,
                                     'accuracy': scores['test_accuracy'].mean(),
                                     'precision': (scores['test_precision'].mean() - 0.75) / 0.25
                                     }], ignore_index=True)

        return svm_cv

    def svm_robustness(self, data="main data"):
        """
        This function aims to check robustness of various SVM models for 60-month investment horizon.
        :param data: string (represents type of data sets to be evaluated ("main data" or "extended data")
        :return: a dataframe contains accuracy, f1, and auc scores for different k-folds and SVM models
        """
        # get necessary input and output
        if data == "main data":
            x_scaled, y = Helper.preprocessing(self.main_60m)
        else:
            x_scaled, y = Helper.preprocessing(self.extended_60m)
        # define variables
        k_fold = [3, 5, 10]
        scoring = ['accuracy', 'f1', 'roc_auc']
        dict_classifiers = {
            'linear': SVC(kernel='linear', C=1, random_state=0),
            'rbf_1.4': SVC(kernel='rbf', gamma=1.4, C=1, random_state=0),
            'rbf_0.4': SVC(kernel='rbf', gamma=0.4, C=1, random_state=0),
            'polynomial': SVC(kernel='poly', degree=3, coef0=1, C=1)
        }
        svm_robustness = pd.DataFrame(columns=[
            'k_fold', 'investment_horizon', 'kernel', 'accuracy', 'f1_score', 'auc'])
        for fold in k_fold:
            for kernel, model_instance in dict_classifiers.items():
                scores = cross_validate(estimator=model_instance,
                                        X=x_scaled,
                                        y=y,
                                        scoring=scoring,
                                        cv=fold,
                                        return_train_score=True)
                sorted(scores.keys())  # see name of available keys of scoring methods
                svm_robustness = svm_robustness.append([{'k_fold': fold,
                                                         'investment_horizon': '60m',
                                                         'kernel': kernel,
                                                         'accuracy': scores['test_accuracy'].mean(),
                                                         'f1_score': scores['test_f1'].mean(),
                                                         'auc': scores['test_roc_auc'].mean()
                                                         }], ignore_index=True)

        return svm_robustness

    def plot_svm_decision_function(self, data="main data"):
        """
        Plot decision function of SVM linear kernel of 60-month investment horizon
        :param data: string (represents type of data sets to plot decision function( "main data" or "extended data")
        """
        # source: https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
        # prepare data for SVM model and plotting
        if data == "main data":
            x_scaled, y = Helper.preprocessing(self.main_60m)
        else:
            x_scaled, y = Helper.preprocessing(self.extended_60m)

        # fit the linear SVM model
        clf_plot = SVC(kernel='linear', C=1, random_state=0)
        clf_plot.fit(x_scaled, y)
        plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        # produce a legend with the unique colors from the scatter
        # source: https://matplotlib.org/3.1.1/gallery/lines_bars_
        fig, ax1 = plt.subplots()
        scatter = ax1.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        legend1 = ax1.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax1.add_artist(legend1)
        # get axes
        ax = plt.gca()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        # create grid to evaluate model
        xx = np.linspace(x_lim[0], x_lim[1], 30)
        yy = np.linspace(y_lim[0], y_lim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf_plot.decision_function(xy).reshape(XX.shape)
        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        ax.set_xlabel("(Standardized) Price-to-Book Ratio")
        ax.set_ylabel("(Standardized) Dividend Yield")
        plt.title("Panel 1: SVM Linear Kernel, Decision Boundary")
        plt.show()

    def svm_platt(self, data="main data"):
        """
        Get probabilistic output (based on Platt's method) of SVM linear kernel of 60-month investment horizon and plot
        :param data: string (represents type of data sets to plot post. probabilities ("main data" or "extended data")
        """
        # source: https://www.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html#buq4yzy-1
        # prepare data for SVM model and plotting
        if data == "main data":
            x_scaled, y = Helper.preprocessing(self.main_60m)
        else:
            x_scaled, y = Helper.preprocessing(self.extended_60m)

        # fit SVM model with probabilistic output
        clf_platt = SVC(kernel='linear', C=1, probability=True, random_state=0)
        clf_platt.fit(x_scaled, y)
        plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, s=30, cmap=plt.cm.Paired)

        # plot probabilistic outputs
        # produce legend
        fig, ax1 = plt.subplots()
        scatter = ax1.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        legend1 = ax1.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax1.add_artist(legend1)
        # get current axes
        ax = plt.gca()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        # create grid to evaluate model
        xx = np.linspace(x_lim[0], x_lim[1], 30)
        yy = np.linspace(y_lim[0], y_lim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf_platt.predict_proba(xy)[:, 0]  # get posterior probabilities for negative events
        Z = Z.reshape(XX.shape)
        # plot probabilities & color_bar
        # source:
        # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.contourf.html#matplotlib.axes.Axes.contourf
        cs = ax.contourf(XX, YY, Z, levels=10, alpha=0.5, linestyles=['-'], cmap='plasma')
        cbar = fig.colorbar(cs, orientation='vertical', ax=ax)
        cbar.ax.set_ylabel("Probability")
        ax.set_xlabel("(Standardized) Price-to-Book Ratio")
        ax.set_ylabel("(Standardized) Dividend Yield")
        plt.title("Panel 2: SVM Linear Kernel, Posterior Probability")
        plt.show()

    def prob_regime(self, data="main data"):
        """
        Define 10 probabilities regimes for SVM linear kernel for 60-month investment horizon and
        plot the 10 regimes overtime
        :param data: string (represents type of data sets to estimate prob. regimes ("main data" or "extended data")
        :return regime: a dataframe matches each data instance of the two explanatory variables to a probability
        regime
        :return regime_realizations: a dataframe consists of total number of realizations in each probability regime
        """

        # run SVM linear kernel with probabilistic output
        if data == "main data":
            x_scaled, y = Helper.preprocessing(self.main_60m)
            regime = self.main_60m
        else:
            x_scaled, y = Helper.preprocessing(self.extended_60m)
            regime = self.extended_60m

        clf_platt = SVC(kernel='linear', C=1, probability=True, random_state=0)
        clf_platt.fit(x_scaled, y)
        # get predicted probabilities for negative events
        negative_events = clf_platt.predict_proba(x_scaled)[:, 0]
        regime['neg_events_prediction'] = negative_events

        # assign posterior probabilities to 10 regimes
        conditions = [(regime['neg_events_prediction'] < 0.1),
                      (0.1 <= regime['neg_events_prediction']) & (regime['neg_events_prediction'] < 0.2),
                      (0.2 <= regime['neg_events_prediction']) & (regime['neg_events_prediction'] < 0.3),
                      (0.3 <= regime['neg_events_prediction']) & (regime['neg_events_prediction'] < 0.4),
                      (0.4 <= regime['neg_events_prediction']) & (regime['neg_events_prediction'] < 0.5),
                      (0.5 <= regime['neg_events_prediction']) & (regime['neg_events_prediction'] < 0.6),
                      (0.6 <= regime['neg_events_prediction']) & (regime['neg_events_prediction'] < 0.7),
                      (0.7 <= regime['neg_events_prediction']) & (regime['neg_events_prediction'] < 0.8),
                      (0.8 <= regime['neg_events_prediction']) & (regime['neg_events_prediction'] < 0.9),
                      (0.9 <= regime['neg_events_prediction'])]
        choices = np.arange(1, 11).tolist()
        regime['prob_regime'] = np.select(conditions, choices)

        # estimate realizations over time for each of the 10 regimes
        regime_realizations = pd.DataFrame([])
        for i in range(1, 11):
            regime_realizations = \
                regime_realizations.append([
                    {'prob_regime': i,
                     'ex_post_probability': (regime.loc[regime['prob_regime'] == i]
                                             ['neg_events_prediction'].mean()) * 100,
                     'total_realizations': len(regime.loc[regime['prob_regime'] == i].index)
                     }], ignore_index=True)

        return regime, regime_realizations

    def plot_prob_regime(self, data="main data"):
        """
        This function plots probability regimes and MSCI World Index values over time of relevant data set
        :param data: string (represents type of data sets to estimate prob. regimes ("main data" or "extended data")
        """
        if data == "main data":
            regime, _ = self.prob_regime(data=data)
            msci_value_plot = self.original_data.loc[
                self.original_data.index <= pd.Timestamp(year=2011, month=8, day=31)].reset_index()
            title = 'Posterior Probability Regimes using Platt Sigmoid Approach, Dec 1974 - Aug 2011'
        else:
            regime, _ = self.prob_regime(data=data)
            msci_value_plot = self.original_data.loc[
                self.original_data.index <= pd.Timestamp(year=2015, month=3, day=31)].reset_index()
            title = 'Posterior Probability Regimes using Platt Sigmoid Approach, Dec 1974 - March 2015'

        # line plot the 10 probability regimes over time
        # source: https://matplotlib.org/gallery/api/two_scales.html
        regime_plot = regime[["prob_regime"]].reset_index()

        fig, ax1 = plt.subplots(figsize=(15, 8))
        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Probability Regimes', color=color)
        ln1 = ax1.plot(regime_plot['Date'], regime_plot['prob_regime'], label='Regime', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('MSCI World TR Index', color=color)  # we already handled the x-label with ax1
        ln2 = ax2.plot(msci_value_plot['Date'], msci_value_plot['Value'], label='MSCI World TR Index', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(title)
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')
        plt.show()


class NeuralNetworks:
    """
    Evaluate MLP models on MSCI World Index 'main data' and 'extended data'
    """

    def __init__(self):

        # define global variables for the class
        self.original_data = Helper.get_data()
        self.main_1m, self.main_12m, self.main_60m, self.extended_1m, self.extended_12m, self.extended_60m = \
            Helper.transform(self.original_data)

    @staticmethod
    def build_model_nn1(score):
        model = models.Sequential()
        model.add(layers.Dense(4, activation='relu', input_shape=(2,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=[score])
        return model

    @staticmethod
    def build_model_nn2(score):
        model = models.Sequential()
        model.add(layers.Dense(4, activation='relu', input_shape=(2,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(12, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=[score])
        return model

    @staticmethod
    def build_model_nn3(score):
        model = models.Sequential()
        model.add(layers.Dense(4, activation='relu', input_shape=(2,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=[score])
        return model

    def nn_cross_validate(self, horizon='60m'):
        """
        This function performs neural networks (NN) 10-fold cross-validation on 'main data' &
        'extended data' of relevant investment horizon.
        Note: running time of the function can be more than 5 minutes for CPU Processor
        :param horizon: string (stands for investment horizon we want to perform: 60m, 12m, or 1m)
        :return a dataframe consists of accuracy and precision score of different NN architectures
        """
        # source:
        # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.7-predicting-house-prices.ipynb

        # get input and output numpy arrays of relevant investment horizon
        if horizon == '60m':
            x_scaled_main, y_main = Helper.preprocessing(self.main_60m)
            x_scaled_extended, y_extended = Helper.preprocessing(self.extended_60m)
        elif horizon == '12m':
            x_scaled_main, y_main = Helper.preprocessing(self.main_12m)
            x_scaled_extended, y_extended = Helper.preprocessing(self.extended_12m)
        else:
            x_scaled_main, y_main = Helper.preprocessing(self.main_1m)
            x_scaled_extended, y_extended = Helper.preprocessing(self.extended_1m)

        # define necessary variables
        neural_networks_cv = pd.DataFrame(columns=['data', 'investment_horizon', 'network', 'metric', 'metric_score'])
        metrics_dict = {'accuracy_rate': 'accuracy',
                        'precision_rate': metrics.Precision()}
        k_fold = 10
        num_epochs = 20
        val_samples_main = len(x_scaled_main) // k_fold
        val_samples_extended = len(x_scaled_extended) // k_fold

        # iterate through each metric for each network on 'main data'
        for score, score_instance in metrics_dict.items():
            networks_dict = {'nn1': self.build_model_nn1(score_instance),
                             'nn2': self.build_model_nn2(score_instance),
                             'nn3': self.build_model_nn3(score_instance)}
            for network, network_instance in networks_dict.items():
                # apply 10-fold cross-validation
                val_scores = []
                for fold in range(k_fold):
                    # Prepare the validation data: data from partition no. k
                    validation_data = x_scaled_main[val_samples_main * fold: val_samples_main * (fold + 1)]
                    validation_targets = y_main[val_samples_main * fold: val_samples_main * (fold + 1)]
                    # Prepare the training data: data from all other partitions
                    train_data = np.concatenate(
                        [x_scaled_main[:fold * val_samples_main], x_scaled_main[(fold + 1) * val_samples_main:]],
                        axis=0)
                    train_targets = np.concatenate(
                        [y_main[:fold * val_samples_main], y_main[(fold + 1) * val_samples_main:]],
                        axis=0)
                    model = network_instance
                    model.fit(train_data, train_targets, epochs=num_epochs, batch_size=4, verbose=0)
                    val_loss, val_metric = model.evaluate(validation_data, validation_targets, verbose=0)
                    val_scores.append(val_metric)
                # record cross-validated result of each metric of each network
                neural_networks_cv = neural_networks_cv.append([{'data': 'main data',
                                                                 'investment_horizon': horizon,
                                                                 'network': network,
                                                                 'metric': score,
                                                                 'metric_score': np.mean(val_scores)
                                                                 }], ignore_index=True)

        # iterate through each metric for each network on 'extended data'
        for score, score_instance in metrics_dict.items():
            networks_dict = {'nn1': self.build_model_nn1(score_instance),
                             'nn2': self.build_model_nn2(score_instance),
                             'nn3': self.build_model_nn3(score_instance)}
            for network, network_instance in networks_dict.items():
                # apply 10-fold cross-validation
                val_scores = []
                for fold in range(k_fold):
                    # Prepare the validation data: data from partition no. k
                    validation_data = x_scaled_extended[val_samples_extended * fold: val_samples_extended * (fold + 1)]
                    validation_targets = y_extended[val_samples_extended * fold: val_samples_extended * (fold + 1)]
                    # Prepare the training data: data from all other partitions
                    train_data = np.concatenate(
                        [x_scaled_extended[:fold * val_samples_extended],
                         x_scaled_extended[(fold + 1) * val_samples_extended:]],
                        axis=0)
                    train_targets = np.concatenate(
                        [y_extended[:fold * val_samples_extended], y_extended[(fold + 1) * val_samples_extended:]],
                        axis=0)
                    model = network_instance
                    model.fit(train_data, train_targets, epochs=num_epochs, batch_size=4, verbose=0)
                    val_loss, val_metric = model.evaluate(validation_data, validation_targets, verbose=0)
                    val_scores.append(val_metric)
                # record cross-validated result of each metric of each network
                neural_networks_cv = neural_networks_cv.append([{'data': 'extended data',
                                                                 'investment_horizon': horizon,
                                                                 'network': network,
                                                                 'metric': score,
                                                                 'metric_score': np.mean(val_scores)
                                                                 }], ignore_index=True)

        neural_networks_cv['adjusted_score'] = np.where(neural_networks_cv['metric'] == 'precision_rate',
                                                        (neural_networks_cv['metric_score'] - 0.75) / 0.25,
                                                        neural_networks_cv['metric_score'])

        return neural_networks_cv

    def nn_learning_curve(self, horizon='60m', nn='nn1'):
        """
        This function aims at plotting learning curves on 'main data'
        for different investment horizons during cross validation,
        which is used to evaluate over-fitting/under-fitting of the model.
        :param nn: string (stands for type of NN we want to perform the plot: 'nn1', 'nn2', or 'nn3')
        :param horizon: string (stands for investment horizon we want to perform: '60m', '12m', or '1m')
        """

        # get input and output numpy arrays of relevant investment horizon
        if horizon == '60m':
            x_scaled_main, y_main = Helper.preprocessing(self.main_60m)
        elif horizon == '12m':
            x_scaled_main, y_main = Helper.preprocessing(self.main_12m)
        else:
            x_scaled_main, y_main = Helper.preprocessing(self.main_1m)

        # build relevant NN
        if nn == 'nn1':
            model = self.build_model_nn1(score='accuracy')
        elif nn == 'nn2':
            model = self.build_model_nn2(score='accuracy')
        else:
            model = self.build_model_nn3(score='accuracy')

        k_fold = 10
        num_epochs = 20
        val_samples_main = len(x_scaled_main) // k_fold
        train_loss_plot = []
        train_acc_plot = []
        val_loss_plot = []
        val_acc_plot = []
        for fold in range(k_fold):
            # Prepare the validation data: data from partition # k
            validation_data = x_scaled_main[val_samples_main * fold: val_samples_main * (fold + 1)]
            validation_targets = y_main[val_samples_main * fold: val_samples_main * (fold + 1)]
            # Prepare the training data: data from all other partitions
            train_data = np.concatenate(
                [x_scaled_main[:fold * val_samples_main], x_scaled_main[(fold + 1) * val_samples_main:]],
                axis=0)
            train_targets = np.concatenate(
                [y_main[:fold * val_samples_main], y_main[(fold + 1) * val_samples_main:]],
                axis=0)
            model_plot = model
            # Train the model
            history = model_plot.fit(train_data, train_targets,
                                     epochs=num_epochs, batch_size=4, verbose=0,
                                     validation_data=(validation_data, validation_targets))

            # append data from each fold
            train_loss = history.history['loss']
            train_acc = history.history['accuracy']
            val_loss = history.history['val_loss']
            val_acc = history.history['val_accuracy']

            train_loss_plot.append(train_loss)
            val_loss_plot.append(val_loss)
            train_acc_plot.append(train_acc)
            val_acc_plot.append(val_acc)

        # compute the average of the per-epoch accuracy/loss scores for all folds:
        average_train_loss = [np.mean([x[i] for x in train_loss_plot]) for i in range(num_epochs)]
        average_val_loss = [np.mean([x[i] for x in val_loss_plot]) for i in range(num_epochs)]
        average_train_acc = [np.mean([x[i] for x in train_acc_plot]) for i in range(num_epochs)]
        average_val_acc = [np.mean([x[i] for x in val_acc_plot]) for i in range(num_epochs)]

        # plot training and validation loss
        plt.plot(range(1, len(average_train_loss) + 1), average_train_loss, 'bo', label='Training loss')
        plt.plot(range(1, len(average_val_loss) + 1), average_val_loss, 'b', label='Val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Training and Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss of ' + horizon + ' Investment Horizon - ' + nn)
        plt.show()

        # plot training and validation accuracy
        plt.clf()  # clear figure
        plt.plot(range(1, len(average_train_acc) + 1), average_train_acc, 'bo', label='Training acc')
        plt.plot(range(1, len(average_val_acc) + 1), average_val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Training and Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy of ' + horizon + ' Investment Horizon - ' + nn)
        plt.show()

    def nn_platt(self, nn='nn1', data='main data'):
        """
        This function plots the posterior probability of 60-month investment horizon on
        'main data' or 'extended data' for different NNs.
        :param data: string (represents type of data sets to estimate prob. regimes ("main data" or "extended data"))
        :param nn: string (stands for type of NN we want to perform the plot ('nn1', 'nn2', or 'nn3'))
        :return: posterior probability plot
        """
        if data == "main data":
            x_scaled, y = Helper.preprocessing(self.main_60m)
        else:
            x_scaled, y = Helper.preprocessing(self.extended_60m)

        if nn == 'nn1':
            model = self.build_model_nn1(score='accuracy')
        elif nn == 'nn2':
            model = self.build_model_nn2(score='accuracy')
        else:
            model = self.build_model_nn3(score='accuracy')
        model.fit(x_scaled, y, epochs=20, batch_size=4)
        plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, s=30, cmap=plt.cm.Paired)

        # plot probabilistic outputs
        # produce legend
        fig, ax1 = plt.subplots()
        scatter = ax1.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        legend1 = ax1.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax1.add_artist(legend1)
        # get current axes
        ax = plt.gca()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        # create grid to evaluate model
        xx = np.linspace(x_lim[0], x_lim[1], 30)
        yy = np.linspace(y_lim[0], y_lim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = 1 - model.predict_proba(xy)  # get posterior probabilities for negative events
        Z = Z.reshape(XX.shape)
        # plot probabilities & color_bar
        # source:
        # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.contourf.html#matplotlib.axes.Axes.contourf
        cs = ax.contourf(XX, YY, Z, levels=10, alpha=0.5, linestyles=['-'], cmap='plasma')
        cbar = fig.colorbar(cs, orientation='vertical', ax=ax)
        cbar.ax.set_ylabel("Probability")
        ax.set_xlabel("(Standardized) Price-to-Book Ratio")
        ax.set_ylabel("(Standardized) Dividend Yield")
        plt.title("Posterior Probability PLot of " + nn + ' on ' + data)
        plt.show()


class Helper:

    @staticmethod
    def get_data():
        """
        Import and re-format original data (provided by professor Viebig)
        """
        cwd = os.path.dirname(os.path.abspath(__file__))
        original_data = pd.read_excel(cwd + '\\DATA_SVM_JV.xlsx', sheet_name='Sheet1')
        original_data = original_data.iloc[1:, :]  # remove first row (which is text)
        original_data.columns = ['Date', 'Value', 'P/B_ratio', 'Dividend_yield']
        original_data['Dividend_yield'] = original_data['Dividend_yield'] / 100  # rescale dividend yield
        original_data = original_data.set_index('Date')
        original_data = original_data.astype(float)

        return original_data

    @staticmethod
    def transform(original_data):
        """
        Transform original data to get 'main data' & 'extended data' set for
        three different investment horizons (from here on, I refer to data in paper
        of Prof. Viebig as 'main data' and the original_data set provided by
        Prof. Viebig as 'extended data').
        :param original_data: data frame contains all data provided by professor Viebig
        :return 'main_data' and 'extended_data' set for three investment horizons
        (in total returns 6 dataframes)
        """

        # plotting to get first impression of two explanatory variables
        # source: https://www.kaggle.com/discdiver/guide-to-scaling-and-standardizing
        # scatter plot
        plt.scatter(original_data['P/B_ratio'], original_data['Dividend_yield'], s=50, cmap='autumn')
        plt.show()
        # distribution plot of un-standardized data
        fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 3))
        sns.kdeplot(original_data['P/B_ratio'], ax=ax1)
        sns.kdeplot(original_data['Dividend_yield'], ax=ax1)
        plt.ylabel('Density')
        plt.title('Distribution Plot of Un-standardized Data')
        plt.show()
        # distribution plot of standardized data
        fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 3))
        sns.kdeplot(
            (original_data['Dividend_yield'] - original_data['Dividend_yield'].mean()
             )/original_data['Dividend_yield'].std(), ax=ax1)
        sns.kdeplot(
            (original_data['P/B_ratio'] - original_data['P/B_ratio'].mean()
             )/original_data['P/B_ratio'].std(), ax=ax1)
        plt.ylabel('Density')
        plt.title('Distribution Plot of Standardized Data')
        plt.show()

        # calculate rolling returns for three investment horizons
        original_data['1_month_return'] = original_data.Value.pct_change(periods=1)
        original_data['12_month_return'] = original_data.Value.pct_change(periods=12)
        original_data['60_month_return'] = original_data.Value.pct_change(periods=60)

        # shift future returns to match current values of P/B ratio & dividend yields
        original_data['60_month_return'] = original_data['60_month_return'].shift(periods=-60)
        original_data['12_month_return'] = original_data['12_month_return'].shift(periods=-12)
        original_data['1_month_return'] = original_data['1_month_return'].shift(periods=-1)

        # separate data of each investment horizons to 'main data' and 'extended_data'
        extended_1m = original_data[["P/B_ratio", "Dividend_yield", "1_month_return"]].dropna()
        extended_12m = original_data[["P/B_ratio", "Dividend_yield", "12_month_return"]].dropna()
        extended_60m = original_data[["P/B_ratio", "Dividend_yield", "60_month_return"]].dropna()

        main_1m = extended_1m.loc[extended_1m.index <= pd.Timestamp(year=2016, month=7, day=29)]
        main_12m = extended_12m.loc[extended_12m.index <= pd.Timestamp(year=2015, month=8, day=31)]
        main_60m = extended_60m.loc[extended_60m.index <= pd.Timestamp(year=2011, month=8, day=31)]

        # define separate return thresholds for 'main data' & 'extended data'
        threshold_1m_extended = extended_1m["1_month_return"].quantile(.25)
        threshold_12m_extended = extended_12m["12_month_return"].quantile(.25)
        threshold_60m_extended = extended_60m["60_month_return"].quantile(.25)

        threshold_1m_main = main_1m["1_month_return"].quantile(.25)
        threshold_12m_main = main_12m["12_month_return"].quantile(.25)
        threshold_60m_main = main_60m["60_month_return"].quantile(.25)

        # add label to returns for each investment horizon of 'main data' & 'extended data'
        extended_1m.loc[extended_1m['1_month_return'] >= threshold_1m_extended, '1_month_label'] = 1
        extended_1m.loc[extended_1m['1_month_return'] < threshold_1m_extended, '1_month_label'] = -1
        extended_12m.loc[extended_12m['12_month_return'] >= threshold_12m_extended, '12_month_label'] = 1
        extended_12m.loc[extended_12m['12_month_return'] < threshold_12m_extended, '12_month_label'] = -1
        extended_60m.loc[extended_60m['60_month_return'] >= threshold_60m_extended, '60_month_label'] = 1
        extended_60m.loc[extended_60m['60_month_return'] < threshold_60m_extended, '60_month_label'] = -1

        main_1m.loc[main_1m['1_month_return'] >= threshold_1m_main, '1_month_label'] = 1
        main_1m.loc[main_1m['1_month_return'] < threshold_1m_main, '1_month_label'] = -1
        main_12m.loc[main_12m['12_month_return'] >= threshold_12m_main, '12_month_label'] = 1
        main_12m.loc[main_12m['12_month_return'] < threshold_12m_main, '12_month_label'] = -1
        main_60m.loc[main_60m['60_month_return'] >= threshold_60m_main, '60_month_label'] = 1
        main_60m.loc[main_60m['60_month_return'] < threshold_60m_main, '60_month_label'] = -1

        # get overview of distribution of returns labels for each investment horizon
        # of 'main data' & 'extended data'
        (extended_1m['1_month_label'].value_counts() / len(extended_1m.index)) * 100
        (extended_12m['12_month_label'].value_counts() / len(extended_12m.index)) * 100
        (extended_60m['60_month_label'].value_counts() / len(extended_60m.index)) * 100

        (main_1m['1_month_label'].value_counts() / len(main_1m.index)) * 100
        (main_12m['12_month_label'].value_counts() / len(main_12m.index)) * 100
        (main_60m['60_month_label'].value_counts() / len(main_60m.index)) * 100

        return main_1m, main_12m, main_60m, extended_1m, extended_12m, extended_60m

    @staticmethod
    def get_summary_statistics(original_data):
        """Get key statistics of the two data samples"""
        result_extended = pd.DataFrame()
        result_extended = result_extended.append(
            original_data[["P/B_ratio", "Dividend_yield"]].describe()).reset_index()
        result_extended = result_extended.append([{'index': 'skewness',
                                                   'P/B_ratio': original_data["P/B_ratio"].skew(),
                                                   'Dividend_yield': original_data["Dividend_yield"].skew()}],
                                                 ignore_index=True)
        result_extended = result_extended.append([{'index': 'kurtosis',
                                                   'P/B_ratio': original_data["P/B_ratio"].kurtosis(),
                                                   'Dividend_yield': original_data["Dividend_yield"].kurtosis()}],
                                                 ignore_index=True)

        main_data = original_data.loc[original_data.index <= pd.Timestamp(year=2016, month=8, day=31)]
        result_main = pd.DataFrame()
        result_main = result_main.append(main_data[["P/B_ratio", "Dividend_yield"]].describe()).reset_index()
        result_main = result_main.append([{'index': 'skewness',
                                           'P/B_ratio': main_data["P/B_ratio"].skew(),
                                           'Dividend_yield': main_data["Dividend_yield"].skew()}],
                                         ignore_index=True)
        result_main = result_main.append([{'index': 'kurtosis',
                                           'P/B_ratio': main_data["P/B_ratio"].kurtosis(),
                                           'Dividend_yield': main_data["Dividend_yield"].kurtosis()}],
                                         ignore_index=True)

        return result_extended, result_main

    @staticmethod
    def preprocessing(data):
        """
        Extract (standardized) input and output variables for relevant investment horizon.
        :param data: dataframe of an investment horizon
        """
        # source: https://scikit-learn.org/stable/modules/preprocessing.html
        x = data.iloc[:, :2].to_numpy()
        y = data.iloc[:, 3].to_numpy()
        x_scaled = (x - x.mean(axis=0)) / x.std(axis=0)

        return x_scaled, y


def main():
    """Control function"""
    try:
        svm = SVM()
        svm_cv = svm.svm_cross_validate(horizon='60m')
        svm_robustness = svm.svm_robustness(data="main data")
        svm.plot_svm_decision_function(data="main data")
        svm.svm_platt(data="main data")
        regime, regime_realizations = svm.prob_regime(data="main data")
        svm.plot_prob_regime(data="extended data")

        nn = NeuralNetworks()
        nn_cv = nn.nn_cross_validate(horizon='60m')
        nn.nn_learning_curve(horizon='60m', nn='nn3')
        nn.nn_platt(nn="nn3", data="main data")

    except BaseException as be:
        print(be)


if __name__ == '__main__':
    main()
