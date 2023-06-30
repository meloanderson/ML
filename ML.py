# File: ML.py
# Import necessary packages
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import deepcopy as cp

from sklearn.base import is_classifier

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Neural Network MLP
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from keras import backend as K

# SVM
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay
)
# Decision Tree

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold)

import ML

import tensorflow as tf

from pathlib import Path
import os.path
from datetime import datetime

from sklearn.decomposition import PCA
from paretochart.paretochart import pareto

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_style("ticks")
sns.set_context("paper")

random_state = 42



from datetime import datetime
import os
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import recall_score, precision_score

class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Keras-based MLP (multi-layer perceptron) classifier
    """

    def __init__(self, **kwargs):
        # Initialize model
        self._estimator_type = 'classifier'
        self.response_method = 'predict_proba'
        self.classes_ = [0, 1]
        self.threshold = 0.5
        self.n_neurons = kwargs.get('n_neurons',2)
        self.input_dim = kwargs.get('input_dim',None)
        self.output_dim = kwargs.get('output_dim',1)
        self.hidden_neurons_activation_function = kwargs.get('hidden_neurons_activation_function', 'tanh')
        self.output_neurons_activation_function = kwargs.get('output_neurons_activation_function', 'sigmoid')
        self.epochs = kwargs.get('epochs', 100)
        self.model = self.build_model()
        self.estimator = self.model

        # Model checkpoint
        self.checkpoint_path = kwargs.get('checkpoint', None)
        self.monitor = kwargs.get('monitor', 'val_loss')
        self.mode = kwargs.get('mode', 'auto')
        self.best_model_checkpoint = None

        if self.checkpoint_path is None:
        # Set a default checkpoint file path if None is provided
            self.checkpoint_path = 'model_checkpoint.h5'

    def build_model(self):
        """
        Build MLP model
        """
        print("self.n_neurons:", self.n_neurons)
        print("self.input_dim:", self.input_dim)
        model = Sequential()
        model.add(Dense(self.n_neurons, input_dim=self.input_dim, activation=self.hidden_neurons_activation_function))
        model.add(Dense(self.output_dim, activation=self.output_neurons_activation_function))
        model.compile(
            optimizer=optimizers.SGD(learning_rate=0.1),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit model using training and validation data
        
        """

        checkpoint_callback = ModelCheckpoint(
            self.checkpoint_path,
            monitor=self.monitor,
            mode=self.mode,
            save_best_only=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint_callback],
            verbose=0,
            epochs=self.epochs
        )

        # Find the best model checkpoint file
        self.best_model_checkpoint = checkpoint_callback.best
        if isinstance(self.best_model_checkpoint, str):
            self.best_model_checkpoint = str(self.best_model_checkpoint)
        elif isinstance(self.best_model_checkpoint, float):
            self.best_model_checkpoint = self.checkpoint_path

        # Load weights if the best model checkpoint is available
        if self.best_model_checkpoint and isinstance(self.best_model_checkpoint, str):
            self.model.load_weights(self.best_model_checkpoint)

        self.history = history

    def predict_proba(self, X):
        """
        Make probability predictions
        """
        y_pred = self.model.predict(X)
        return y_pred

    def predict(self, X):
        """
        Make binary predictions
        """
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba > self.threshold).astype(int)

    def get_sensitivity(self, X, y):
        """
        Compute sensitivity/recall score
        """
        y_pred = self.predict(X)
        return recall_score(y, y_pred)

    def get_precision(self, X, y):
        """
        Compute precision score
        """
        y_pred = self.predict(X)
        return precision_score(y, y_pred)
    



def create_classifier(x,column,classifier):
    if x[column]<=classifier:
        return 0
    else:
        return 1
    
def get_PCA_comp(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca= PCA(n_components=11)
    pca.fit(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    var = pca.explained_variance_ratio_.cumsum()

    for i in range(len(var)):
        if var[i]>=0.8:
            num_comp = i
            fig, ax = plt.subplots(figsize=(11, 5))
            pareto(pca.explained_variance_ratio_)
            ax.grid();
            break

    return var, var_ratio, num_comp

def plot_distribution_base(data):
    num_plots = data.shape[1]
    num_rows = (num_plots + 3) // 4
    fig = plt.figure(figsize=(12, num_rows*2))

    for i, column in enumerate(data.columns):
        ax = fig.add_subplot(num_rows, 4, i+1)

        sns.histplot(data[column], ax=ax)
        
        ax.grid()
        ax.set_title(f'{column} Distribution')
        ax.set_xlabel(f'{column}')
        ax.set_ylabel('Count')

    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    plt.tight_layout()
    plt.show()

def log_transform(X):
    X_log= np.log(X)
    return X_log

def normalize_L1_L2(X_Train, X_Test, L_type):
    transformer = preprocessing.Normalizer(norm=L_type)
    normalized_Xtrain = transformer.transform(X_Train)
    normalized_Xtest = transformer.transform(X_Test)
    return normalized_Xtrain, normalized_Xtest

def get_train_test(X, y, test_size, random_state,**kwargs):
    """
    Method for training multiple models
    """
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X.values,
                                                        y.values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y,
                                                        **kwargs)

    return X_train_cv,X_test_cv, y_train_cv,y_test_cv

def interpolation(fpr, tpr):
    interp_fpr = np.linspace(0, 1, 100)
    interp_tpr = np.interp(interp_fpr, fpr, tpr)
    interp_tpr[0] = 0.
    return interp_fpr, interp_tpr

def train(X, X_test, y, y_test,model_klass, n_splits=5, n_init=1, use_pca=False, tag="", transformation="",random_seed=None, **kwargs):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_seed)
    #Several empty lists are initialized to store the results for each fold:


    f1_score_val_list   = []
    f1_score_train_list = []
    fprs_list           = [] # fprs_list: List to store the false positive rates.
    tprs_list           = [] # tprs_list: List to store the true positive rates.
    auc_list            = [] # auc_list: List to store the area under the curve (AUC) scores.
    scaler_list         = [] # scaler_list: List to store the StandardScaler objects used for feature scaling.
    model_list          = []

    X_train_scaled_list = [] # Initialize list to store scaled training data
    y_train_list        = [] # Initialize list to store scaled training data
    X_train_list        = [] # Initialize list to store scaled training data

    precision_list      = []
    sensitivity_list    = []
    pcas                = []
    best_pca = 'Not Applicable'
    num_comp_list       = []

    # Create the figure and axes for plotting the ROC curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Validação cruzada só em Training Data
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

        X_train = X[train_idx, :]
        y_train = y[train_idx]

        X_val = X[val_idx, :]
        y_val = y[val_idx]

        # Escala
        scaler = StandardScaler()
        scaler_list.append(scaler)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        #PCA
        if use_pca:
            var, var_ratio, num_comp = get_PCA_comp(X_train)
            pca = PCA(n_components=num_comp)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)
            print(f"The number of components required to explain 80% of the variance for this model was {num_comp}")

            # Update the model's input_dim parameter if it exists in the kwargs
            if 'input_dim' in kwargs:
                kwargs['input_dim'] = num_comp

        X_train_scaled_list.append(X_train_scaled)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        
        
        f1_score_val=0
        model = None

        for idx in range(n_init):

            _model = model_klass(**kwargs)
            _model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

            _y_pred = _model.predict(X_train_scaled)
            _y_pred_val = _model.predict(X_val_scaled)

            _f1_score_val = f1_score(y_val, _y_pred_val, average='weighted')
            if _f1_score_val > f1_score_val:
                y_pred_val = _y_pred_val
                y_pred = _y_pred
                model = _model
            K.clear_session()

        sensitivity = 100 *  recall_score(y_val, y_pred_val)
        precision= 100 *  precision_score(y_val, y_pred_val)

        precision_list.append(precision)
        sensitivity_list.append(sensitivity)

        print(f"========================= FOLD {fold} ==========================")

        print(f"The results of this train's F1-Score is {f1_score(y_train, y_pred, average='weighted'):.2}")
        print(f"The results of this validation's F1-Score is {f1_score(y_val, y_pred_val,average='weighted'):.2}")

        f1_score_val_list.append(f1_score(y_val, y_pred_val, average='weighted'))
        f1_score_train_list.append(f1_score(y_train, y_pred, average='weighted'))
        model_list.append(model)

        if use_pca:
            pcas.append(pca)
            num_comp_list.append(num_comp)
        

        y_hat_val = model.predict_proba(X_val_scaled)

        viz = RocCurveDisplay.from_predictions(
            y_val,
            y_hat_val,
            ax = ax,
            alpha=0.3,
            lw=1
        )


        interp_fpr, interp_tpr = interpolation(viz.fpr, viz.tpr)
        fprs_list.append(interp_fpr)
        tprs_list.append(interp_tpr)
        auc_list.append(viz.roc_auc)

    mean_fpr = np.mean(fprs_list, axis=0)
    mean_tpr = np.mean(tprs_list, axis=0)
    mean_auc = np.mean(auc_list)
    std_auc  = np.std(auc_list)
    mean_val = np.mean(f1_score_val_list)
    std_val  = np.std(f1_score_val_list)
    mean_pre = np.mean(precision_list, axis=0)
    mean_sen = np.mean(sensitivity_list, axis=0)
    std_pre  = np.std(precision_list)
    std_sen  = np.std(precision_list)
    mean_f1  = np.mean(f1_score_val_list, axis=0)
    std_f1   = np.std(f1_score_val_list)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color='blue',
        lw=2,
        label=r"Mean ROC (AUC = %.2f $\pm$ %.2f)" %(mean_auc, std_auc)
    )

    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            color='g',
            ls=":",
            lw=0.5)
    ax.legend()

    if fold == n_splits-1:
        print(f"The average F1-Score of the train set is {np.mean(f1_score_train_list): .2} +- {np.std(f1_score_train_list): .2} ")
        print(f"The average F1-Score of the validation set is é {mean_val: .2} +- {std_val: .2} ")
        best_model_idx = np.argmax(f1_score_val_list)
        print(f"The best fold was: {best_model_idx} ")
        best_model  = model_list[best_model_idx]
        best_scaler = scaler_list[best_model_idx]
    

        # Fazer a inferência em Test Data
   
        X_test_scaled       = best_scaler.transform(X_test)
        if use_pca:
            best_pca        = pcas[best_model_idx]
            X_test_scaled   = best_pca.transform(X_test_scaled)
            best_num_comp = num_comp_list[best_model_idx]
            

        y_pred_test         = best_model.predict(X_test_scaled)
        X_train_scaled_best = X_train_scaled_list[best_model_idx]
        X_train_best        = X_train_list[best_model_idx]
        y_train_best        = y_train_list[best_model_idx]

        print(f"The F1-Score for the test data is: {f1_score(y_test, y_pred_test):.2} with the best model")

        print('===============================Summary of analysis====================================')
        if use_pca:
            print(f'The amount of components required to explain 80% variance of the best model is {best_num_comp}')
        print(f"The Average F1-Score of this test is {mean_f1:.2f} %")
        print(f"The Deviation of F1-Score is: {std_f1:.2}")
        print(f"The Average sensitivity of this test is {mean_sen:.2f} %")
        print(f"The Deviation of sensitivity is: {std_sen:.2} ")
        print(f"The Average Precision of this model is { mean_pre:.2f} %")
        print(f"The Deviation of Precision is: {std_pre:.2} ")
        print(f"The Average accuracy of this model is {mean_auc:.2f} %")
        print(f"The Deviation of accuracy is: {std_auc:.2} ")
        
    return {
        'model': best_model,
        'kwargs': str(kwargs),
        'scaler': best_scaler,
        'X_train': X_train_best,
        'X_train_scaled': X_train_scaled_best,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,  # Add X_test_scaled to the returned dictionary
        'y_train': y_train_best,
        'y_test': y_test,
        'mean_val': mean_val,
        'std_val': std_val,
        'pca_used': use_pca,
        'pca': best_pca,
        'transformation': transformation,
        'tag': tag
    }

def roc_auc_models(model_name_class):
    num_rows = (len(model_name_class) + 1) // 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, num_rows*6))
    axs = axs.flatten()

    for i, model in enumerate(model_name_class):
        model_name = model.model_name
        ax = axs[i]
        # row = i // 2
        # col = i % 2
        # ax = axs[row, col]

        model.plot_roc_train(ax=ax)
        model.plot_roc_test(ax=ax, color='orange')
        
        ax.grid()
        ax.set_title(f'{model_name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(['Train', 'Test'])

    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    plt.tight_layout()
    plt.show()

def get_validation_info(model_name_class):

    model_dicts_storage = []
    model_pca_dicts_storage = []

    # Iterate over each model instance
    for i, model in enumerate(model_name_class):
        model_type = model.model_name
        Parameters = model.kwargs
        validation_f1 = model.mean_val
        validation_f1_deviation = model.std_val 
        precision = model.get_precision()
        sensitivity = model.get_sensitivity()

    
        if model_type.endswith('_pca'):
            model_pca_dict = {
                'Model': model_type[:-4],
                # 'Parameters': Parameters,
                'Precision PCA': precision,
                'Sensitivity PCA':sensitivity,
                'Validation F1 PCA': validation_f1,
                'Validation F1 deviation PCA': validation_f1_deviation,
            }
            model_pca_dicts_storage.append(model_pca_dict)
            
        else:
            model_dict = {
                            'Model': model_type,
                            # 'Parameters': Parameters,
                            'Precision': precision,
                            'Sensitivity': sensitivity,
                            'Validation F1': validation_f1,
                            'Validation F1 deviation': validation_f1_deviation,
                        }
            model_dicts_storage.append(model_dict)

    model_no_pca = pd.DataFrame(model_dicts_storage)
    model_pca = pd.DataFrame(model_pca_dicts_storage)

    # Merge dataframes and keep all rows
    results = model_no_pca.merge(model_pca, how='outer')

    return results, model_no_pca, model_pca

def plot_final_results(results_final):
    sns.set_style("ticks")
    sns.set_context("talk")
    fig, ax = plt.subplots(2, 1, figsize=(16, 16))
    
    # Plot without PCA analysis
    ax[0].errorbar(range(results_final.shape[0]),
                   results_final['Validation F1'],
                   results_final['Validation F1 deviation'])
    ax[0].grid(True)
    sns.despine(offset=5, ax=ax[0])

    ax[0].set_title("Models performance in the validation set -- No PCA Analysis")
    ax[0].set_ylabel("F1 Score")
    ax[0].set_xlabel("Model")
    ax[0].set_xticks(range(results_final.shape[0]))
    ax[0].set_xticklabels(results_final['Model'], rotation=90)
    
    # Plot with PCA analysis
    ax[1].errorbar(range(results_final.shape[0]),
                   results_final['Validation F1 PCA'],
                   results_final['Validation F1 deviation PCA'])
    ax[1].grid(True)
    sns.despine(offset=5, ax=ax[1])

    ax[1].set_title("Models performance in the validation set PCA")
    ax[1].set_ylabel("F1 Score")
    ax[1].set_xlabel("Model")
    ax[1].set_xticks(range(results_final.shape[0]))
    ax[1].set_xticklabels(results_final['Model'], rotation=90)
    
    plt.tight_layout()
    plt.show()

class ModelResults:
    def __init__(self, results):
        self.model = results['model']
        self.kwargs = results['kwargs']
        self.using_pca = results['pca_used']
        self.transformation = results['transformation']
        self.tag = results['tag']
        concat_string = '_model'+ str(self.tag) + str(self.transformation)
        if self.using_pca: concat_string += '_pca'
        self.model_name = self.model.__class__.__name__ + concat_string
        self.scaler = results['scaler']
        self.X_train = results['X_train']
        self.X_train_scaled = results['X_train_scaled']
        self.X_test = results['X_test']
        self.X_test_scaled = results['X_test_scaled']
        self.y_train = results['y_train']
        self.y_test = results['y_test']
        self.mean_val = results['mean_val']
        self.std_val = results['std_val']
        self.pca = results['pca']
        self.y_pred = None  # Initialize y_pred attribute as None

    def plot_hist(self, estimator_name : str = "test", **kwargs):
        self.y_pred = self.model.predict(self.X_test_scaled)
        plt.hist(self.y_pred)
        plt.xlabel('Predictions')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {self.model_name} Predictions')
        plt.show()
    
    def plot_roc(self, X, y, estimator_name: str = "train", **kwargs):
        y_hat = self.model.predict_proba(X)
        
        # Check if y_hat has only one column, if so, create a dummy column of zeros
        if y_hat.shape[1] == 1:
            zeros = np.zeros_like(y_hat)
            y_hat = np.concatenate([zeros, y_hat], axis=1)

        fpr, tpr, thresholds = roc_curve(y, y_hat[:, 1], pos_label=1)
        auc_score = auc(fpr, tpr)
        return RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=auc_score,
            estimator_name=estimator_name
        ).plot(**kwargs)

    def plot_roc_train(self, **kwargs):
        self.plot_roc(self.X_train_scaled, self.y_train, estimator_name="train", **kwargs)

    def plot_roc_test(self, **kwargs):
        self.plot_roc(self.X_test_scaled, self.y_test, estimator_name="test", **kwargs)

    def plot_distribution(self, X, y, ax=None, estimator_name: str = "train", **kwargs):
        y = pd.Series(y)  # Convert y to pandas Series
        y_hat = self.model.predict_proba(X)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sns.distplot(y_hat[y.values == 1, 1], label="Good", ax=ax)
        ax.set_xlim([0, 1])
        sns.distplot(y_hat[y.values == 0, 1], label="Bad", ax=ax)
        ax.legend()
        return ax

    def plot_distribution_train(self, **kwargs):
        self.plot_distribution(self.X_train_scaled, self.y_train, estimator_name="train", **kwargs)

    def plot_distribution_test(self, **kwargs):
        self.plot_distribution(self.X_test_scaled, self.y_test, estimator_name="test", **kwargs)

    def correlation_matrix_train(self, **kwargs):
        self.y_pred = self.model.predict(self.X_train_scaled)
        cm = confusion_matrix(self.y_train, self.y_pred)
        ax = sns.heatmap(cm, cmap ="BuGn", annot=True, fmt='g')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Bad Wine(0)', 'Good Wine(1)'])
        ax.set_yticklabels(['Bad Wine(0)', 'Good Wine(1)'])

    def correlation_matrix_test(self, **kwargs):
        self.y_pred = self.model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, self.y_pred)
        ax = sns.heatmap(cm, cmap ="BuGn", annot=True, fmt='g')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Bad Wine(0)', 'Good Wine(1)'])
        ax.set_yticklabels(['Bad Wine(0)', 'Good Wine(1)'])
    
    def get_sensitivity(self, **kwargs):
        self.y_pred = self.model.predict(self.X_test_scaled)
        return (f" {100 *  recall_score(self.y_test, self.y_pred):.4f} %")

    def get_precision(self, **kwargs):
        self.y_pred = self.model.predict(self.X_test_scaled)
        return (f" {100 *  precision_score(self.y_test, self.y_pred):.4f} %")
    
