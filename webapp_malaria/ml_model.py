import pandas as pd
import numpy as np
import pickle
from matplotlib.axis import YTick
from sklearn import preprocessing

df = pd.read_csv("outbreak_detect.csv")

df.dropna(inplace=True)

#data processing

#labelencoding
LE=preprocessing.LabelEncoder()
#fitting it to our dataset
df.Outbreak = LE.fit_transform(df.Outbreak)

#method 2 to load the data in the form of arrays -by library numpy
import numpy as np
X = np.array(df[['avgHumidity',	'Rainfall',	'Positive',	'pf']])
Y = np.array(df[['Outbreak']])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,Y_train)

y_pred = model.predict(sc.transform(X_test))
print(y_pred)

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print("sucess loaded")
