
'''Lets the user choose an MLModel and returns it.
Must choose 'DT', 'GB', 'KN', 'LC', 'LR' '''

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

class MLModel:

	def getClass(self):
		return self.classifier

	def __init__(self, choice):

		if (type(choice) is not str) or (choice not in ['DT', 'GB', 'KN', 'LC', 'LR']):
			raise Exception('Must choose \'DT\', \'GB\', \'KN\', \'LC\', \'LR\'')

		self.choice = choice

		if self.choice is 'DT':
			self.classifier = DecisionTreeClassifier(max_depth=6)

		if self.choice is 'GB':
			self.classifier = GaussianNB()

		if self.choice is 'KN':
			self.classifier = KNeighborsClassifier(n_neighbors=10)

		if self.choice is 'LC':
			self.classifier = LinearSVC()

		if self.choice is 'LR':
			self.classifier = LogisticRegression()


	def __str__(self):
		return '{self.choice}'.format(self=self)

	def __repr__(self):
		return 'MLModel({self.choice})'.format(self=self)
