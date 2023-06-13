import pandas
import numpy 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'



bank_data = pandas.read_csv("ukol_04_data.csv")

data_categorical = ["job", "marital", "education", "default","housing","loan", "contact", "campaign", "poutcome" ]

data_numeric = ["age","balance", "duration", "pdays", "previous"]



bank_numeric= bank_data[data_numeric].to_numpy()

y = bank_data["y"] 
encoder = OneHotEncoder()

encoded_categorical=encoder.fit_transform(bank_data[data_categorical])
encoded_categorical = encoded_categorical.toarray()

X =numpy.concatenate([encoded_categorical, bank_numeric],axis=1)


#print(y.value_counts())
#no     43922
# yes    5810
#encoder=LabelEncoder()
#y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=4)
clf=clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data, filled=True, feature_names= list(encoder.get_feature_names_out())+ data_numeric,class_names=["no_acccount", "yes_account"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('tree.png') 

# matice záměn 

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=clf.classes_,)
#plt.show()

# metriky 
accuracy = accuracy_score(y_test, y_pred)
#print (accuracy)
# 0.9018096514745308 - celková přesnost modelu 

# Pokud jde o to, aby banka kontaktovala ty, kteří mohou mít zájem s tím, že z možných strategií preferuje tu, kdy raději  opomine někoho, 
# kdo zájem má, byť byl modelem nesprávně zařazen do kategorie těch, kteří zájem nemají, než aby kontaktovala ty, co, zájem nemají,
#  i když byly modelem označeni za ty, co ho mají -  znamená to, že raději nekontaktuje ty, kdo jsou falešně negativní,
#  než by obvolávala i ty, kteří jsou falešně pozitivní. - tedy vybíráme metriku, která penalizuje falešně pozitivní případy- čili precision 

precision = precision_score(y_test, y_pred, pos_label="yes")
print (precision)
# 0.6481927710843374 - tedy asi 65% přesnost

#  pro výšku stromu v rozmezí 5 - 12 (jen precision_score, bez ukládání stromů do obrázku)

dp = range(5,13)
precision_scores =[]
for i in dp: 
    clf = DecisionTreeClassifier(random_state=42, max_depth=i)
    clf=clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    precision_scores.append(precision_score(y_test, y_pred,pos_label="yes"))
print(precision_scores)
# [0.6172199170124482, 0.6162324649298597, 0.5968280467445742, 0.6221786064769381, 0.5817737998372661, 0.5806174957118353, 0.5731319554848967, 0.5523076923076923]
# žádný z vetších hloubek stromu  nepřináší lepší výsledek, než výše spočítáná hloubka 4 

# algoritmus K Nearest Neighbour


encoder = OneHotEncoder()
encoded_categorical=encoder.fit_transform(bank_data[data_categorical])
encoded_categorical = encoded_categorical.toarray()

encoder=LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
bank_numeric = scaler.fit_transform(bank_data[data_numeric])

X =numpy.concatenate([encoded_categorical, bank_numeric], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

model_1= KNeighborsClassifier()
params_1={"n_neighbors":range(3,27,4)}

# vytvoření klasifikátoru: 
clf_1=GridSearchCV(model_1, params_1, scoring="precision")
clf_1.fit(X,y)
#print(clf_1.best_params_)
#print(clf_1.best_score_)

#{'n_neighbors': 19}
#0.5619353442724035


# Support Vector Machine 

clf=LinearSVC()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
#plt.show()
precision = precision_score(y_test, y_pred, pos_label=1)
print (precision) 

# precision score = 0.661354581673306; - 66%, tedy cca o 2% lepší výsledek než v případě rozhodovacího stromu a výrazně lepší 
#výsledek než u KNN (o témeř 10%) 



