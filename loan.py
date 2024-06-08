
import pandas as pd
import numpy as np


#ld=pd.read_csv('/workspaces/loan_defaulter/Loan_Defaulters - Copy.csv')
ld=pd.read_csv('Loan_Defaulters - Copy.csv')
ld=ld.drop(['ID'],axis=1)

ld['Verification Status']=ld['Verification Status'].replace('Source Verified','Verified')

#Outliers
q1=np.percentile(ld['Funded Amount Investor'],25,method='midpoint')
q2=np.percentile(ld['Funded Amount Investor'],50,method='midpoint')
q3=np.percentile(ld['Funded Amount Investor'],75,method='midpoint')
IQR=q3-q1
low_lim=q1-1.5*IQR
up_lim=q3+1.5*IQR
ld['Funded Amount Investor']=ld['Funded Amount Investor'].clip(lower=low_lim,upper=up_lim)


q1=np.percentile(ld['Revolving Balance'],25,method='midpoint')
q2=np.percentile(ld['Revolving Balance'],50,method='midpoint')
q3=np.percentile(ld['Revolving Balance'],75,method='midpoint')
IQR=q3-q1
low_lim=q1-1.5*IQR
up_lim=q3+1.5*IQR
ld['Revolving Balance']=ld['Revolving Balance'].clip(lower=low_lim,upper=up_lim)

q1=np.percentile(ld['Total Accounts'],25,method='midpoint')
q2=np.percentile(ld['Total Accounts'],50,method='midpoint')
q3=np.percentile(ld['Total Accounts'],75,method='midpoint')
IQR=q3-q1
low_lim=q1-1.5*IQR
up_lim=q3+1.5*IQR
ld['Total Accounts']=ld['Total Accounts'].clip(lower=low_lim,upper=up_lim)

q1=np.percentile(ld['Total Received Late Fee'],25,method='midpoint')
q2=np.percentile(ld['Total Received Late Fee'],50,method='midpoint')
q3=np.percentile(ld['Total Received Late Fee'],75,method='midpoint')
IQR=q3-q1
low_lim=q1-1.5*IQR
up_lim=q3+1.5*IQR
ld['Total Received Late Fee']=ld['Total Received Late Fee'].clip(lower=low_lim,upper=up_lim)

q1=np.percentile(ld['Total Collection Amount'],25,method='midpoint')
q2=np.percentile(ld['Total Collection Amount'],50,method='midpoint')
q3=np.percentile(ld['Total Collection Amount'],75,method='midpoint')    
IQR=q3-q1
low_lim=q1-1.5*IQR
up_lim=q3+1.5*IQR
ld['Total Collection Amount']=ld['Total Collection Amount'].clip(lower=low_lim,upper=up_lim)

#Label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

ld=ld.drop(['Sub Grade','Batch Enrolled'],axis=1)

ld['Grade']=le.fit_transform(ld['Grade'])
ld['Employment Duration']=le.fit_transform(ld['Employment Duration'])
ld['Verification Status']=le.fit_transform(ld['Verification Status'])
ld['Loan Title']=le.fit_transform(ld['Loan Title'])
ld['Initial List Status']=le.fit_transform(ld['Initial List Status'])
ld['Application Type']=le.fit_transform(ld['Application Type'])

#Test & train data
y=ld['Loan Status']
X=ld.drop(['Loan Status'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25)

#model random forest
from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier()
rf_clf.fit(X_train,y_train)
y_pred=rf_clf.predict(X_test)

#create pickle
import pickle
with open('model1.pkl','wb') as model1_file:  #wb-- write as binary
    pickle.dump(rf_clf,model1_file) 

