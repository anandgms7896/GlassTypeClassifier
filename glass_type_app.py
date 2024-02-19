# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,ConfusionMatrixDisplay 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache_data()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data()

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
klist=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]
def prediction(model,klist):
    glass_type=model.predict([klist])[0]
    if glass_type == 1:
      return "building windows float processed"
    elif glass_type == 2:
      return "building windows non float processed" 
    elif glass_type == 3:
      return "vehicle windows float processed"  
    elif glass_type == 4:
      return "vehicle windows non float processed"  
    elif glass_type==5:
      return "containers" 
    elif glass_type==6:
      return  "tableware" 
    else:
      return "headlamp"

st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis") 
if st.sidebar.checkbox("Show Raw Data") :
  st.subheader("Glass Type Data set" )  
  st.dataframe(glass_df) 

st.sidebar.subheader("Scatter plot")
st.set_option('deprecation.showPyplotGlobalUse', False)
features_list=st.sidebar.multiselect("Select the x-axis values:",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for i in features_list:
    st.subheader(f"scatter plot between {i} and glass_type")
    plt.figure(figsize=(10,5))
    plt.scatter(glass_df[i],glass_df["GlassType"])
    st.pyplot()



st.sidebar.subheader("Visualisation Selector")
plot_types=st.sidebar.multiselect('Select the Charts/Plots:',('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))
if 'Histogram' in plot_types:
  # plot histogram
   st.sidebar.subheader("Histogram")

   features_list=st.sidebar.selectbox("Select the x-axis values for histogram:",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
   st.subheader(f"histogram for {features_list}")
   plt.figure(figsize=(10,5))
   plt.hist(glass_df[features_list],bins="sturges")
   st.pyplot()
if 'Box Plot' in plot_types:
    features_list=st.sidebar.multiselect("Select the x-axis values for boxplot:",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    st.subheader(f"boxplot  for {features_list}")
    plt.figure(figsize=(10,5))
    sns.boxplot(x=glass_df[features_list])
    st.pyplot() 
if 'Count Plot' in plot_types:
  # plot count plot
    st.subheader(f"countplot  ")
    plt.figure(figsize=(10,5))
    sns.countplot(x=glass_df["GlassType"])
    st.pyplot() 
if 'Pie Chart' in plot_types:
  
  # plot pie chart
    st.subheader(f"piechart ")
    plt.figure(figsize=(10,5))
    plt.pie(glass_df["GlassType"].value_counts())
    st.pyplot() 
if 'Correlation Heatmap' in plot_types:
  # plot correlation heatmap
    st.subheader(f"correlation heatmap  ")
    plt.figure(figsize=(10,5))
    sns.heatmap(glass_df.corr(),annot=True)
    st.pyplot() 
if 'Pair Plot' in plot_types:
  # plot pair plot 
    st.subheader(f"plotpair ")
    plt.figure(figsize=(10,5))
    sns.pairplot(glass_df)
    st.pyplot()     

ri=st.sidebar.slider("input RI", float(glass_df["RI"].min()),float(glass_df["RI"].max()))
na=st.sidebar.slider("input Na", float(glass_df["Na"].min()),float(glass_df["Na"].max())) 
mg=st.sidebar.slider("input Mg", float(glass_df["Mg"].min()),float(glass_df["Mg"].max())) 
al=st.sidebar.slider("input Al", float(glass_df["Al"].min()),float(glass_df["Al"].max())) 
si=st.sidebar.slider("input Si", float(glass_df["Si"].min()),float(glass_df["Si"].max())) 
k=st.sidebar.slider("input K", float(glass_df["K"].min()),float(glass_df["K"].max())) 
ca=st.sidebar.slider("input Ca", float(glass_df["Ca"].min()),float(glass_df["Ca"].max())) 
ba=st.sidebar.slider("input Ba", float(glass_df["Ba"].min()),float(glass_df["Ba"].max())) 
fe=st.sidebar.slider("input Fe", float(glass_df["Fe"].min()),float(glass_df["Fe"].max())) 
st.sidebar.subheader("Choose Classifier")
classifier=st.sidebar.selectbox("Classifier",('Support Vector Machine', 'Random Forest Classifier','LogisticRegression'))
if classifier == 'Support Vector Machine':
  st.sidebar.subheader("Model Hyperparameters")
  c_value=st.sidebar.number_input("enter c value",1,100,step=1)
  kernel_type=st.sidebar.radio("select the kernel",("linear","rbf","poly"))
  gamma=st.sidebar.number_input("enter gamma",1,100,step=1)
    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
  if(st.sidebar.button("classify")):
    svc_model=SVC(C = c_value, kernel = kernel_type, gamma = gamma).fit(X_train,y_train)
    svc_model.score(X_train,y_train) 
    y_pred=svc_model.predict(X_test)
    glass_type=prediction(svc_model,[ri, na, mg, al, si, k, ca, ba, fe])
    st.write(f"the type of glass predicted is{glass_type} ")
    st.write(confusion_matrix(y_test,y_pred))
if classifier == 'Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)

    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf,[ ri, na, mg, al, si, k, ca, ba, fe])
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        y_pred=rf_clf.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
        st.pyplot()
        # S1.1: Implement Logistic Regression with hyperparameter tuning
if classifier == 'LogisticRegression':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("c_value", 1, 100, step = 1)
    max_iter_input = st.sidebar.number_input("Maximum iter", 10, 100, step = 10)

    if st.sidebar.button('Classify'):
        st.subheader("LogisticRegression")
        lr_clf = LogisticRegression(C = c_value, max_iter = max_iter_input)
        lr_clf.fit(X_train,y_train)
        accuracy = lr_clf.score(X_test, y_test)
        glass_type = prediction(lr_clf, [ri, na, mg, al, si, k, ca, ba, fe])
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        y_pred=lr_clf.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
        st.pyplot()