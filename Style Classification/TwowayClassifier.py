import readFile
from sklearn import tree
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.svm import SVC

#random forest for two-way classifier

features_X, labels_X, image_Name = readFile.generate_targetFeature('feature.xlsx')

skf = StratifiedKFold(labels_X, 5)

def train_model(model):
    labels = []
    predict_labels = []
    predict_prob = []
    wrong_index = []
    for train, test in skf:
        features_train = []
        labels_train = []
        features_test = []
        labels_test = []

        for index in train:
            features_train.append(features_X[index])
            labels_train.append(labels_X[index])
        for index in test:
            features_test.append(features_X[index])
            labels_test.append(labels_X[index])

        clf = model(n_estimators=1000)
        clf = clf.fit(features_train, labels_train)
        predict_labels_test = clf.predict(features_test)
        #predict_labels_prob = clf.predict_proba(features_test)

        for label in labels_test:
            labels.append(label)
        for label in predict_labels_test:
            predict_labels.append(label)
       # for prob in predict_labels_prob:
       #     predict_prob.append(prob)

        for i in range(0, len(labels_test)):
            if labels_test[i] != predict_labels_test[i]:
                wrong_index.append(test[i])

    return labels, predict_labels, wrong_index

def print_result(model_name, labels, predict_labels):
    print model_name
    print 'Result:' + '(precision, recall)'
    print str(accuracy(labels, predict_labels))
    print ''

def printEstimatorScores(model, X_test, y_test):
	"""
	Calculate & Print F1 Score for classifier

	:param model: Classifier
	:param X_test: test samples
	:param Y_test: test labels
	"""
	y_predict = model.predict(X_test)
	print "F1 Score:", f1_score(y_test, y_predict, average='weighted')
	print

def printF1CVScore(scores):
	"""
	Print F1 scores from K Fold Cross Validation

	:param scores: k-fold scores
	"""
	print "F1 CV Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
	print

def accuracy(labels, predict_labels):
    if len(labels) != len(predict_labels):
        print 'Error accuracy()'
        return None

    attribute_map = {}
    total_positive = 0.
    true_positive = 0.
    false_positive = 0.
    for i in range(0, len(labels)):
        if labels[i] != -1:
            total_positive += 1
        if predict_labels[i] != -1 and predict_labels[i] == labels[i]:
            true_positive += 1
        if predict_labels[i] != -1 and predict_labels[i] != labels[i]:
            false_positive += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / total_positive
    return precision, recall



def random_forest(X, y, cv=5):
	"""
	Implement & Fit Random Forest Classifier. Then perform k fold cross-validation.

	:param X: sample data
	:param y: label data
	:cv cross-validation generator
	"""
	rfModel = ensemble.RandomForestClassifier(n_estimators=1000)
	f1_scores = cross_val_score(rfModel, X, y, cv=cv, scoring='f1_weighted')
	print rfModel
	printF1CVScore(f1_scores)
	X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2)
	print 'RandomForestClassifier is Fitting'
	rfModel.fit(X_train, y_train)
	printEstimatorScores(rfModel, X_test, y_test)
	return rfModel





#knnModel = knn(features_X, labels_X, cv=5)
#rfModel = random_forest(features_X, labels_X, cv=5)
#svModel = svm(features_X, labels_X, cv=5)
#loModel = logistic_regress(features_X, labels_X, cv=5)
#naive = naiveBayes(features_X, labels_X, cv=5)


#
# # Random Forest
labels, predict_labels, wrong_index = train_model(ensemble.RandomForestClassifier)
print_result('Random Forest', labels, predict_labels)
#print_wrong_data(wrong_index)
#
rfModel = random_forest(features_X, labels_X, cv=5)
#



X_train, X_test, y_train, y_test = train_test_split(features_X,labels_X,test_size=0.20, random_state=40)
clf = OneVsRestClassifier(LinearSVC(random_state=0))
LinearSVC(C=0.01, penalty="l1", dual=False)
y_pred = clf.fit(X_train, y_train).predict(X_test)
target_names = ['Vangogh', 'Gauguin', 'Braque', 'Gris', 'Monet', 'Raphael', 'Titian']
print(classification_report(y_test, y_pred, target_names=target_names))