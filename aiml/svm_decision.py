#svm

from sklearn import svm

# Training data
X = [[150,7],[170,8],[130,6],[140,6.5]]
y = ['Apple','Apple','Orange','Orange']

# Create SVM model
model = svm.SVC(kernel='linear')
model.fit(X,y)

# Prediction
print(model.predict([[160,7]]))




# Decision Tree


from sklearn.tree import DecisionTreeClassifier
# Training data
X = [[35,50000],[22,20000],[30,35000],[25,18000]]
y = ['Yes','No','Yes','No']
# Create model
model = DecisionTreeClassifier()
model.fit(X,y)
# Prediction
print(model.predict([[28,30000]]))