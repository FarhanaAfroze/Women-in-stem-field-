import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns

st.markdown("WOMEN IN STEM FIELD")

img=Image.open('/Users/shifa/Downloads/stem.jpeg')
st.image(img)

st.markdown("PROBLEM STATEMENT")
st.markdown("This dataset is from the Pew Research Center 2017 Stem Survey. This dataset was made for analyzing US adults' careers and their education in the STEM field. It has a sample size of 4914 adults and all of these samples were taken from all 50 U.S states and The District of Columbia. Our project topic is about women in the STEM field. We would like to analyze how women are doing in the stem field, both in school and the workplace. We believe this dataset can be used to answer several relevant questions. Do more women pursue STEM majors in university than men? Do women tend to pursue a STEM major or a non-STEM major? Do women like Math and Science courses more than non-STEM courses? In the job sector do they get less appreciated? Do they face harassment in this field more? Do men get more value than women even though both genders have similar qualifications and play the same role in the workplace? In conclusion, we would like to address issues like these. Also, We would like to analyze this dataset and would like to answer all these questions")

st.markdown('Our raw dataset')
raw_data = pd.read_csv('/Users/shifa/Documents/raw_data.csv')

st.subheader('Raw data information')
st.dataframe(raw_data)



st.markdown('After cleaning and converting into categorcal to numercal our new dataset looks: ')
data = pd.read_csv('/Users/shifa/Documents/eda_survey.csv')

st.subheader('Clean dataset information')

st.dataframe(data)

infos = data.info()
st.write(infos)


st.sidebar.markdown("## SELECTION ")

st.sidebar.markdown("Click to explore dataset")

st.sidebar.subheader('EXPLORE')

if st.sidebar.checkbox('Look to the data'):
	st.subheader('Look to the data')
	st.write(data.head())
if st.sidebar.checkbox('Click columns'):
	st.subheader('All columns')
	cols = data.columns.to_list()
	st.write(cols)
if st.sidebar.checkbox('Missing Values?'):
    st.subheader('Missing values')
    st.write(data.isnull().sum())
    
data.rename(columns = {'START_AGE': 'AGE-1_CATEGORY', 'END_AGE': 'AGE-2_CATEGORY'}, inplace = True)
if st.sidebar.checkbox('statistics'):
	st.subheader('statistics')
	stat = data[['INCOME_AVERAGE','AGE-1_CATEGORY','AGE-2_CATEGORY']].describe()
	st.write(stat)


st.markdown('VISUALIZATION-')

p = sns.catplot(x = "REASON1D", kind = "count", height=5, data = data)
st.pyplot(p)

q = sns.catplot(x = "HARASS1", kind = "count", height=5, data = data)
st.pyplot(q)


st.sidebar.subheader('Visualization(count-plot)')
if st.sidebar.checkbox('count plot'):
	st.subheader('count plot')
	plots = st.sidebar.selectbox('pick column to plot', data.columns)
	hue_plots = st.sidebar.selectbox('categorical variable for hue', data.columns.insert(0,None))
	figure = sns.catplot(x= plots, kind="count", hue =hue_plots , data=data)
	st.pyplot(figure)


st.sidebar.subheader('Visualization(box-plot)')
if st.sidebar.checkbox('Box plot'):
	st.subheader('Box plot')
	plo = st.sidebar.selectbox('pick x column to plot', data.columns)
	plos = st.sidebar.selectbox('pick y column to plot', data.columns)
	figure1 = sns.catplot(x= plo, y = plos, kind="box", data=data)
	st.pyplot(figure1)



#algorithm

pick_model = st.sidebar.selectbox("Pick the ML model", ["NONE" ,"Naive Bayes", "Decision Tree Classifier", "K-means Clustering", "K-nearest neighbour", "Logistic regression"])

st.markdown('ALGORITHM IMPLEMENTATION-')

data_new = pd.read_csv('/Users/shifa/Documents/algo_survey.csv')

features = data_new.iloc[:, [19,20,22,23,24,25,26,12,15,6,7,8]]  

label = data_new.iloc[:, 21]  

st.sidebar.subheader("Features and Label(set-1)")

if st.sidebar.checkbox('Look to the features data'):
	st.subheader('Look to the feature selected for algorithm implementation')
	st.write(features.head())

if st.sidebar.checkbox('Look at the feature label data'):
	st.subheader('Look to the label selected for algorithm implementation')
	st.write(label.head())


st.markdown('What is the tech3 label represent')
st.markdown('tech3 is trying to find out discrimination against women is major, minor or not problem')
st.markdown('Tech3 has 3 classes- Major problem(1), Minor problem(2) and Not problem(3)')
	


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.25, random_state = 50)  


#def user_input():
	#num = st.sidebar.selectbox("Pick any number for the KNN parameters", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

#get_input  = user_input()
#st.subheader('User Input for KNN: ')
#st.write(get_input)



from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

if(pick_model == 'Naive Bayes'):
	model = GaussianNB()

	#Train the model using the training sets
	model.fit(x_train, y_train)

	#Predict the response for test dataset
	prediction = model.predict(x_test)

	score = metrics.accuracy_score(y_test, prediction)
	
	st.text("Accuracy we found for naive bayes model-")
	st.write(score)
	
	reports = metrics.classification_report(y_test, prediction)
	
	st.text("Classification report we got")
	st.write(reports)
	
	matrix = confusion_matrix(y_test, prediction)
	st.text("confusion matrix we got")
	st.write(matrix)
	

	
	
elif(pick_model == 'Logistic regression'):
	logis = LogisticRegression()
	logis.fit(x_train, y_train)
	pred_log = logis.predict(x_test)
	pred_log
	scores = metrics.accuracy_score(y_test, pred_log)
	
	st.text("Accuracy we found for logistic regression model-")
	st.write(scores)
	
	report = metrics.classification_report(y_test, pred_log)
	
	st.text("Classification report we got")
	st.write(report)
	
	matrix1 = confusion_matrix(y_test, pred_log)
	st.text("confusion matrix we got")
	st.write(matrix1)
	
	
	
elif(pick_model == 'K-means Clustering'):
	prints = 'This algorthm is not suitable for this case'
	st.write(prints)

elif(pick_model == 'K-nearest neighbour'):
	printing = 'Wrong algorthm'
	st.write(printing)

elif(pick_model == "Decision Tree Classifier"):
	tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=4)
	tree_model = tree_model.fit(x_train, y_train)
	predictions = tree_model.predict(x_test)
	scoring = metrics.accuracy_score(y_test, predictions)
	
	st.text("Accuracy we found for decision tree classifier model-")
	st.write(scoring)
	
	rep = metrics.classification_report(y_test, predictions)
	st.text("classification report we got")
	st.write(rep)
	
	matrix2 = confusion_matrix(y_test, predictions)
	st.text("confusion matrix we got")
	st.write(matrix2)
	
	

# New dataset for algorithms

#algorithm

pick_choice = st.sidebar.selectbox("Pick another ML model", ["none" ,"Naive-Bayes", "Decision-Tree-Classifier", "K-means-Clustering", "K-nearest-neighbour", "Logistic-regression"])

feature_new = data_new.iloc[:, [3,4,9,12,13,14,15,16,19,20,21,22,23,24,25,26,27]] 

label_new = data_new.iloc[:, 17]

st.sidebar.subheader("Features and Label(set-2)")

if st.sidebar.checkbox('Look to the another features data'):
	st.subheader('Look to the feature selected for algorithm implementation')
	st.write(feature_new.head())

if st.sidebar.checkbox('Look at the another label data'):
	st.subheader('Look to the label selected for algorithm implementation')
	st.write(label_new.head())



st.markdown('What is the REASON1F label represent')
st.markdown('REASON1F trying to find out if women are less interested in science, tech and math and is it major, minor or not problem')
st.markdown('REASON1F has 3 classes- Major reason(1), Minor reason(2) and Not a reason(3)')
	

from sklearn.model_selection import train_test_split
x_trains, x_tests, y_trains, y_tests = train_test_split(feature_new, label_new, test_size = 0.25, random_state = 50)  



from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

if(pick_choice == 'Naive-Bayes'):
	models = GaussianNB()

	#Train the model using the training sets
	models.fit(x_trains, y_trains)

	#Predict the response for test dataset
	predict = models.predict(x_tests)

	score4 = metrics.accuracy_score(y_tests, predict)
	
	st.text("Accuracy we found for naive bayes model-")
	st.write(score4)
	
	reports = metrics.classification_report(y_tests, predict)
	
	st.text("Classification report we got")
	st.write(reports)
	
	matrix6 = confusion_matrix(y_tests, predict)
	st.text("confusion matrix we got")
	st.write(matrix6)
	
	
elif(pick_choice == 'Logistic-regression'):
	logiss = LogisticRegression()
	logiss.fit(x_trains, y_trains)
	pred_logs = logiss.predict(x_tests)
	
	score5 = metrics.accuracy_score(y_tests, pred_logs)
	
	st.text("Accuracy we found for logistic regression model-")
	st.write(score5)
	
	report4 = metrics.classification_report(y_tests, pred_logs)
	
	st.text("Classification report we got")
	st.write(report4)
	
	matrix7 = confusion_matrix(y_tests, pred_logs)
	st.text("confusion matrix we got")
	st.write(matrix7)
	
	
	
elif(pick_choice == 'K-means-Clustering'):
	prints = 'This algorthm is not suitable for this case'
	st.write(prints)
	

elif(pick_choice == 'K-nearest-neighbour'):
	
	knn_model = KNeighborsClassifier(n_neighbors= 5)
	knn_model.fit(x_trains, y_trains)
	pred_knn = knn_model.predict(x_tests)
	meas = metrics.accuracy_score(y_tests, pred_knn)
	
	st.text("Accuracy we found for kNN-")
	st.write(meas)
	
	report55 = metrics.classification_report(y_tests, pred_knn)
	
	st.text("classification report we got")
	st.write(report55)

	matrix9 = confusion_matrix(y_tests, pred_knn)
	st.text("confusion matrix we got")
	st.write(matrix9)

elif(pick_choice == "Decision-Tree-Classifier"):
	tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=4)
	tree_model = tree_model.fit(x_trains, y_trains)
	predict6 = tree_model.predict(x_tests)
	scoring5 = metrics.accuracy_score(y_tests, predict6)
	
	st.text("Accuracy we found for decision tree classifier model-")
	st.write(scoring5)
	
	rep1 = metrics.classification_report(y_tests, predict6)
	st.text("classification report we got")
	st.write(rep1)
	
	matrix8 = confusion_matrix(y_tests, predict6)
	st.text("confusion matrix we got")
	st.write(matrix8)
	


