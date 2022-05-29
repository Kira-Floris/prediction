import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from PIL import Image
import joblib

from utils import get_encoded_label

# importing dataset
dataset = pd.read_csv('./datasets/Clean_Dataset.csv')
dataset_labeled = pd.read_csv('./datasets/label_encoded_dataset.csv')
model = joblib.load('./models/flight_ticket_price_model.sav')
correlation_image = Image.open('correlation_white.png')


# creating header
st.header("Flight Price Prediction AI")

# functions
def description_function():
	st.subheader('Introduction to Flight Ticket Price Prediction')
	st.write('Rwanda Civil Aviation Authority is a regulator for all non military and non state aviations for both private and commercial.')
	st.write('In commercial air transport, it deals with the fairness of tickets for journeys, which raised this project.')

	st.subheader('Project Statement')
	st.write('This project will be able to predict a fair price for tickets depending on past air plane tickets.')
	st.write('the machine learning type to be used is regression to calculate the price with feature selections.')



# analysis function
def analysis_function():
	# import libraries
	st.text('importing libraries')
	st.code(
		'''import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns'''
		)
	# displaying data
	st.subheader('Sample of Data about flights')
	st.code(
			'''
dataset = pd.read_csv('./datasets/Clean_Dataset.csv')
dataset.head(100)
			'''
		)
	st.dataframe(dataset.head(100))

	# checking for null values in dataset
	st.subheader('checking for null values in dataset')
	st.code('''
		dataset.isnull().sum()
		''')
	st.text(dataset.isnull().sum())

	# displaying data in relation to airlines
	st.title('some data exploration on the data')


	st.subheader('Airline relation to duration and price')
	# dataset range slider
	st.text('range for data selection')
	st.code('''
values = st.slider('Select a range for data review for airline selection', 0, int(dataset['flight'].count()), (int(dataset['flight'].count()/4),int(dataset['flight'].count()*(3/4))), key='airlines')
st.text(values)

fig = px.scatter(dataset[values[0]:values[1]], x='duration', y='price',
	size='duration', color='airline', hover_name='airline', log_x=True, size_max=60)
fig.update_layout(width=800)
		''')
	values = st.slider('Select a range for data review for airline selection', 0, int(dataset['flight'].count()), (int(dataset['flight'].count()/4),int(dataset['flight'].count()*(3/4))), key='airlines')
	st.text(values)

	fig = px.scatter(dataset[values[0]:values[1]], x='duration', y='price',
		size='duration', color='airline', hover_name='airline', log_x=True, size_max=60)
	fig.update_layout(width=800)
	st.write(fig)

	# displaying data in relation to classes
	st.subheader('Classes relation to duration and price')
	# dataset range slider
	st.code('''
values_classes = st.slider('Select a range for data review for airline selection', 0, int(dataset['flight'].count()), (int(dataset['flight'].count()/4),int(dataset['flight'].count()*(3/4))), key='classes')
st.text('range for data selection')
st.text(values_classes)

fig_ = px.scatter(dataset[values_classes[0]:values_classes[1]], x='duration', y='price',
	size='duration', color='class', hover_name='class', log_x=True, size_max=60)
fig_.update_layout(width=800)
st.write(fig_)
		''')
	values_classes = st.slider('Select a range for data review for airline selection', 0, int(dataset['flight'].count()), (int(dataset['flight'].count()/4),int(dataset['flight'].count()*(3/4))), key='classes')
	st.text('range for data selection')
	st.text(values_classes)

	fig_ = px.scatter(dataset[values_classes[0]:values_classes[1]], x='duration', y='price',
		size='duration', color='class', hover_name='class', log_x=True, size_max=60)
	fig_.update_layout(width=800)
	st.write(fig_)

	# correlation display
	st.title('Attributes relation to each other')
	st.subheader('data exploration and correlation')
	st.text('''some columns contains categorical values, we need to encode those columns into numerical labels''')
	st.code('''
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
dataset_copy = dataset
dataset_copy['airline'] = encoder.fit_transform(dataset_copy['airline'])
dataset_copy['flight'] = encoder.fit_transform(dataset_copy['flight'])
dataset_copy['source_city'] = encoder.fit_transform(dataset_copy['source_city'])
dataset_copy['departure_time'] = encoder.fit_transform(dataset_copy['departure_time'])
dataset_copy['stops'] = encoder.fit_transform(dataset_copy['stops'])
dataset_copy['arrival_time'] = encoder.fit_transform(dataset_copy['arrival_time'])
dataset_copy['destination_city'] = encoder.fit_transform(dataset_copy['destination_city'])
dataset_copy['class'] = encoder.fit_transform(dataset_copy['class'])
dataset_copy.head(100)
		''')
	st.dataframe(dataset_labeled.head(100))
	st.text('after encoding them, we can them apply correlation to see which columns affect each other')
	st.code('''
fig = (18,8)
plt.figure(figsize=fig)
heatmap = sns.heatmap(dataset_copy.corr(method='pearson'), annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='inferno', linewidths=1, linecolor='Black')
heatmap.set_title('correlation heatmap between variables')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
		''')
	st.image(correlation_image, caption='Correlation Image of variables')

	# displaying selected features for training
	st.subheader('Sample of Data about flights with selected features')
	st.code('''
dataset_selected_features = dataset[['airline','flight','stops','class','duration','price']]
dataset_selected_features.head()
		''')
	st.dataframe(dataset[['airline','flight','stops','class','duration','price']][:100])

	# splitting into training and testing
	st.subheader('splitting the data into training and testing')
	st.code('''
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=0)
		''')

	# training the model
	st.subheader('training our model')
	st.text('import models for training')
	st.code('''
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# importing metrics
from sklearn.metrics import r2_score
		''')

	# linear regression
	st.subheader('linear regression model')
	st.code('''
model1 = LinearRegression()
model1.fit(xtrain, ytrain)

print("intercept", model1.intercept_)
print("coefficients", model1.coef_)

# predicting
ypred1 = model1.predict(xtest)
ypredtrain1 = model1.predict(xtrain)

# evaluation of model
print("evaluation on tests",r2_score(ytest, ypred1))
print("evaluation on train",r2_score(ytrain, ypredtrain1))
		''')
	st.text('results')
	st.code('''
intercept 48048.28010507011
coefficients [ 9.38650084e+02  2.05533389e-01 -3.16754828e+03 -4.45688691e+04
  1.05705442e+02]
evaluation on tests 0.8986374739952657
evaluation on train 0.8979609139938076
		''')

	# decision tree regression
	st.subheader('decision tree regression')
	st.code('''
model3 = DecisionTreeRegressor()
model3.fit(xtrain, ytrain)

# predicting
ypred3 = model3.predict(xtest)
ypredtrain3 = model3.predict(xtrain)

# evaluation of model
print("evaluation on test",r2_score(ytest, ypred3))
print("evaluation on train",r2_score(ytrain, ypredtrain3))
		''')
	st.text('results')
	st.code('''
evaluation on test 0.9748911655693738
evaluation on train 0.9785270668450166
		''')

	# polynomial regression
	st.subheader('polynomial regression')
	st.code('''
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
xtrainpoly = poly.fit_transform(xtrain)
xtestpoly = poly.fit_transform(xtest)

model4 = LinearRegression()
model4.fit(xtrainpoly, ytrain)

# predicting
ypred4 = model4.predict(xtestpoly)
ypredtrain4 = model4.predict(xtrainpoly)

# evaluation of model
print("evaluation on test",r2_score(ytest, ypred4))
print("evaluation on train",r2_score(ytrain, ypredtrain4))
		''')
	st.text('results')
	st.code('''
evaluation on test 0.9261220101656936
evaluation on train 0.9248278173277713
		''')

	# saving the highest performing model
	st.subheader('Saving the model')
	st.text('after evaluating all 3 models, we save the best performing model, which is model3, DecisionTreeRegressor()')
	st.code('''
# saving using pickle
import pickle
pickle.dump(model3, open('models/flight_ticket_price_model.pkl','wb'))

# saving using joblib
import joblib
joblib.dump(model3, open('models/flight_ticket_price_model.sav','wb'))
		''')

	


# artificial intelligence page
def ai_function():
	st.title('AI form for price prediction for flights')
	
	inputs = dict()
	labels = dict()

	with st.form("ai form"):

		airline = st.selectbox('choose airline',(set(dataset['airline'])))
		flight = st.selectbox('choose flight id', (set(dataset['flight'])))
		stops = st.selectbox('choose a number of stops on journey', set(dataset['stops']))
		class_ = st.selectbox('Choose class for the ticket',(set(dataset['class'])))
		duration = st.number_input('enter the duration of flight')

	    # submition form
		submitted = st.form_submit_button("Submit")
		if submitted:
		    inputs['airline'] = airline
		    inputs['flight'] = flight
		    inputs['stops'] = stops
		    inputs['class'] = class_
		    inputs['duration'] = duration

		    # labelled values
		    labels['airline'] = get_encoded_label(airline,'airline', dataset, dataset_labeled)
		    labels['flight'] = get_encoded_label(flight,'flight', dataset, dataset_labeled)
		    labels['stops'] = get_encoded_label(stops,'stops',dataset, dataset_labeled)
		    labels['class'] = get_encoded_label(class_,'class', dataset, dataset_labeled)
		    labels['duration'] = duration

		    # predicting for the model
		    prediction = model.predict([[labels['airline'],labels['flight'],labels['stops'],labels['class'],labels['duration']]])
		    st.subheader('Predicted Price')
		    st.subheader(int(prediction[0]))



	











# creating sidebar menu
with st.sidebar:
	selected = option_menu(
			menu_title = 'Menu',
			options = ['Project Description','Analysis & Codes', 'AI', 'About Me']
		)

if selected == 'Project Description':
	st.title('Project Description')
	description_function()

if selected == 'Analysis & Codes':
	st.title('Codes about Flight Prediction')
	analysis_function()

if selected == 'AI':
	ai_function()

if selected == 'About Me':
	st.title('About Me')


