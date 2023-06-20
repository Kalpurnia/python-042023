import pandas
import numpy 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC 
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


fitness = pandas.read_csv("bodyPerformance.csv")

fitness_categorical = ["gender"]

fitness_numeric = ["age","height_cm", "weight_kg", "body fat_%", "diastolic", "systolic","gripForce"]

fitness_numeric_2 = ["age","height_cm", "weight_kg", "body fat_%", "diastolic", "systolic","gripForce", "sit-ups counts"]

fitness_data= fitness[fitness_numeric].to_numpy()

fitness_data_2 = fitness[fitness_numeric_2].to_numpy()

y = fitness["class"] 
encoder = OneHotEncoder()
encoded_categorical=encoder.fit_transform(fitness[fitness_categorical])
encoded_categorical = encoded_categorical.toarray()

X =numpy.concatenate([encoded_categorical, fitness_data],axis=1)

XX = numpy.concatenate([encoded_categorical, fitness_data_2 ],axis=1)

"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf=clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data, filled=True, feature_names= list(encoder.get_feature_names_out())+ fitness_numeric,class_names=["A", "B", "C", "D"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('fitnesstree.png') 

# matice záměn 

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=clf.classes_,)
#plt.show()
#Správně - tedy zařazeni do kondice A a mají kondici A bylo 605 

# metriky 
accuracy = accuracy_score(y_test, y_pred)
#print (accuracy)
# Accuracy - tedy přesnost predikce tohto rozhodovacího stromu - 0.43280238924838227, tedy asi 43% 

# klasifikace do více tříd - hodí se SVC - , můžeme zkusit One-to-Rest nebo One-to-One 


encoder=LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)


model=SVC(kernel="linear")
params = {"decision_function_shape":["ovo", "ovr"]}
clf = GridSearchCV(model, params,scoring="accuracy")
clf.fit(X,y)

print(clf.best_params_)
print(clf.best_score_)

# {'decision_function_shape': 'ovo'}
# accuracy = 0.4563572342739327
# v tomto případě tedy přináší o něco větší přesnost agoritmus Support Vector Machine, varianta one-to-one"""

# -----------------------
# s přidáním cviku: 

XX_train, XX_test, y_train, y_test = train_test_split(XX, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf=clf.fit(XX_train, y_train)
y_pred=clf.predict(XX_test)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data, filled=True, feature_names= list(encoder.get_feature_names_out())+ fitness_numeric_2,
class_names=["A", "B", "C", "D"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('fitnesstreewithsits_up.png') 

# matice záměn s přidáním cviku 

ConfusionMatrixDisplay.from_estimator(clf, XX_test, y_test, display_labels=clf.classes_,)
plt.show()
#Správně - tedy správbě zařazeni do kondice A a (mají kondici A ) bylo s přidáním cviku  738 testovaných osob 



# metriky s přidáním cviku 
accuracy = accuracy_score(y_test, y_pred)
print (accuracy)
# Accuracy - tedy přesnost predikce tohto rozhodovacího stromu - 0.5027376804380289, tedy s přidáním cviku sit-ups counts 
# se přesnost predikce zvýšila na 50% 

# klasifikace do více tříd - opět  SVC -  One-to-Rest nebo One-to-One 

encoder=LabelEncoder()
y = encoder.fit_transform(y)

XX_train, XX_test, y_train, y_test = train_test_split(XX, y, test_size=0.3, random_state=42, stratify = y)


model=SVC(kernel="linear")
params = {"decision_function_shape":["ovo", "ovr"]}
clf = GridSearchCV(model, params,scoring="accuracy")
clf.fit(XX,y)

print(clf.best_params_)
print(clf.best_score_)

#{'decision_function_shape': 'ovo'}- opět tedy algoritmus Support Vector Machine, varianta one-to-one přináší 
# přesnější predikci, s přidáním cviku se přesnost zvýšila na cca 54% 
#0.5435681946352859