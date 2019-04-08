import pandas as pd
import numpy as np
import random
import math
import itertools

from sklearn.manifold import TSNE
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from scipy.misc import comb
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')





data_frame1 = pd.DataFrame({0 : [np.nan]})
train_set = data_frame1.copy()
test_set = data_frame1.copy()


def normalize(data):
	filter_df = data.copy()
	for i in filter_df.columns:
		if filter_df[i].dtype == 'int64':
			filter_df[i] = np.float64(filter_df[i])
	std_scale = preprocessing.MinMaxScaler().fit(data)
	df_std = std_scale.transform(data)
	k = 0
	print df_std
	for i in filter_df.columns:
		for j in range(len(data)):
			filter_df[i].values[j] = df_std[j][k]
		k += 1
	return filter_df	

def show_correlation():
	data_frame1 = train_set
	co_relationFile = data_frame1.corr(method = 'pearson', min_periods=1)
	co_relationFile.to_csv('co_relationFile.csv')
	new_correlation = pd.read_csv("co_relationFile.csv",header=None)
	print
	print "Correlation between all the columns of data"
	print new_correlation
	

def split_dataset(train_set,test_set):
	indices = train_set.columns[:-1]

	X_train = train_set[indices]
	Y_train = train_set[len(train_set.columns)-1]
	X_train = normalize(X_train)

	indices2 = test_set.columns[:-1]

	X_test = test_set[indices2]
	Y_test = test_set[len(test_set.columns)-1]
	X_test = normalize(X_test)
	return X_train, Y_train, X_test, Y_test

def create_sets():
	
	train_file1 = pd.read_csv('train_file_new1.csv', header=None)
	test_file1 = pd.read_csv('test_file_new1.csv', header=None)

	print "Training Set."
	print train_file1
	print ".................................................................."
	print "Test Set"
	print test_file1
	return train_file1, test_file1

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def random_forest():
	X = X_train
	y = Y_train
	rf_model = RandomForestClassifier(n_estimators=100, # Number of trees
								  max_features=6,    # Num features considered
								  oob_score=True, n_jobs = -1, random_state = 1)
	rf_model.fit(X, y.values.ravel())

	test_preds = rf_model.predict(X_test)
	print "Accuracy of Random Forest is", rf_model.score(X_test, Y_test)*100, "%"
	the_labels = [1,2,3,4,5,6,7]
	p = confusion_matrix(Y_test, test_preds,labels=the_labels)
	#plot_confusion_metrics(p)
	plot_confusion_matrix(p, classes=the_labels,
						  title='Confusion matrix Random Forest')
	


def kNN_classification():
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(X_train, Y_train)
	test_preds = neigh.predict(X_test)
	print "Accuracy of knn ", neigh.score(X_test,Y_test)*100, "%"

	the_labels = [1,2,3,4,5,6,7]
	p = confusion_matrix(Y_test, test_preds,labels=the_labels)
	plot_confusion_matrix(p, classes=the_labels,
						  title='Confusion matrix kNN')

def gaussian_svm():
	clf = svm.SVC(C=10000,gamma = 0.05)
	clf.fit(X_train, Y_train)  
	test_preds = clf.predict(X_test)
	print accuracy_score(Y_test,test_preds)*100
	the_labels = [1,2,3,4,5,6,7]
	p = confusion_matrix(Y_test, test_preds,labels=the_labels)
	#plot_confusion_metrics(p)
	plot_confusion_matrix(p, classes=the_labels,
						  title='Confusion matrix')
def neural_network():
	X = np.array(X_train)
	y = Y_train

	# Get dimensions of input and output
	dimof_input = X.shape[1] 
	dimof_output = np.max(y) + 1
	print('dimof_input: ', dimof_input)
	print('dimof_output: ', dimof_output)

	# Set y categorical
	y = np_utils.to_categorical(y, dimof_output)
	print "yayy ",y
	# Set constants
	batch_size = 128
	dimof_middle = 100
	dropout = 0.2
	countof_epoch = 100
	verbose = 0
	print('batch_size: ', batch_size)
	print('dimof_middle: ', dimof_middle)
	print('dropout: ', dropout)
	print('countof_epoch: ', countof_epoch)
	print('verbose: ', verbose)
	
	model = Sequential()
	model.add(Dense(dimof_middle, input_dim=dimof_input, init='uniform', activation='tanh'))
	model.add(Dropout(dropout))
	model.add(Dense(dimof_middle, init='uniform', activation='tanh'))
	model.add(Dropout(dropout))
	model.add(Dense(dimof_middle, init='uniform', activation='tanh'))
	model.add(Dropout(dropout))
	model.add(Dense(dimof_output, init='uniform', activation='softmax'))
	opt = SGD(lr=0.01)
	model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])

	model.fit(
    X, y,
    validation_split=0.2,
    nb_epoch=500,verbose=0)

	# Evaluate
	loss, accuracy = model.evaluate(X, y, verbose=verbose)
	print('loss: ', loss)
	print('accuracy: ', accuracy)
	print()

	X_new = np.array(X_test)
	Y_new = np_utils.to_categorical(Y_test, dimof_output)
	score = model.evaluate(X_new,Y_new)
	print
	print score[1]*100,"%"
	


def grid_Search():
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
					 'C': [1, 10, 100, 1000]}]
	scores = ['precision', 'recall']
	clf = GridSearchCV(svm.SVC(C=100), tuned_parameters, cv=5,
					   scoring='%s_macro' % 'precision')
	clf.fit(X_train, Y_train)
	return clf.best_params


def cross_validation():
	clf = svm.SVC(C=1000,gamma = 0.05)
	scores = cross_val_score(clf, X_test, Y_test, cv=5)
	print scores


def adaBoost():
	ada = AdaBoostClassifier(base_estimator = RandomForestClassifier(n_estimators = 2, criterion = 'entropy'), 
						 algorithm = 'SAMME.R')
	ada.fit(X_train, Y_train)
	predictions = ada.predict(X_test)
	score = ada.score(X_test,Y_test)
	print 'Accuracy:', "%.2f" %(score*100)

def stacking_voting():
	X = np.array(X_train)
	y = np.array(Y_train)
	neigh = KNeighborsClassifier(n_neighbors=3)
	rf_model = RandomForestClassifier(n_estimators=100, # Number of trees
								  max_features=6,    # Num features considered
								  oob_score=True, n_jobs = -1, random_state = 1)
	eclf1 = VotingClassifier(estimators=[('knn', neigh), ('rf', rf_model)], voting='hard')
	eclf1 = eclf1.fit(X, y)
	score = eclf1.score(X_test,Y_test)
	print score*100


train_set, test_set = create_sets()
X_train, Y_train, X_test, Y_test = split_dataset(train_set, test_set)

options = {
		  1 : show_correlation,
		  2 : grid_Search,
		  3 : kNN_classification,
		  4 : random_forest,
		  5 : gaussian_svm,
		  6 : neural_network,
		  7 : cross_validation,
		  8 : stacking_voting,
		  9 : adaBoost,
		  

}

print "Forest CoverType Dataset"
print "Operations"

print "1. View Correlation between different columns"
print "2. Grid Search"
print "3. kNN Classification for Forest Cover"
print "4. Random Forest Classification for Forest Cover"
print "5. Gaussian kernel SVM for classifying Forest Cover"
print "6. Neural Network Classification for Forest Cover"
print "7. Cross Validation on the Test Dataset"
print "--------------------------------------"
print " Ensembling Techniques"
print "8. Stacking For Classification"
print "9. AdaBoosting for Classification"





print "enter your choice of Operations"
choice = input()	
options[choice]()

