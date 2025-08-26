# Jason Brownlee's "first machine learning project"
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Completed by Zed Lyons as part of the project-based-learning github repository



# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load data (using the commonly-used iris dataset)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


# exploratory data summary
"""
# dimensions of dataset in (rows, columns) or (instances,attributes) in ML jargon
print(dataset.shape)
# head of data
print(dataset.head(20))
# statistical summary e.g. mean, std dev, count, etc.
print(dataset.describe())
# species (class) distribution 
print(dataset.groupby('class').size())
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()
# scatter plot matrix to look for correlations
scatter_matrix(dataset)
plt.show()
"""


#split data into training- and validation-data
