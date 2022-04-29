import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from PIL import Image
import joblib

# importing dataset
dataset = pd.read_csv('./datasets/Clean_Dataset.csv')
model = joblib.load('./models/flight_ticket_price_model.sav')
correlation_image = Image.open('correlation_white.png')


# creating header
st.header("Flight Price Prediction AI")

# functions
# analysis function
def analysis_function():
	# displaying data
	st.title('Sample of Data about flights')
	st.dataframe(dataset[:100])

	# correlation display
	st.title('Attributes relation to each other')
	st.image(correlation_image, caption='Correlation Image of variables')

	# displaying selected features for training
	st.title('Sample of Data about flights with selected features')
	st.dataframe(dataset[['airline','flight','stops','class','duration','price']][:100])

	# displaying data in relation to airlines
	# dataset range slider
	values = st.slider('Select a range for data review for airline selection', 0, int(dataset['flight'].count()), (int(dataset['flight'].count()/4),int(dataset['flight'].count()*(3/4))), key='airlines')
	st.text('range for data selection')
	st.text(values)
	
	st.title('Airline relation to duration and price')
	fig = px.scatter(dataset[values[0]:values[1]], x='duration', y='price',
		size='duration', color='airline', hover_name='airline', log_x=True, size_max=60)
	fig.update_layout(width=800)
	st.write(fig)

	# displaying data in relation to classes
	# dataset range slider
	values_classes = st.slider('Select a range for data review for airline selection', 0, int(dataset['flight'].count()), (int(dataset['flight'].count()/4),int(dataset['flight'].count()*(3/4))), key='classes')
	st.text('range for data selection')
	st.text(values_classes)
	
	st.title('Classes relation to duration and price')
	fig_ = px.scatter(dataset[values_classes[0]:values_classes[1]], x='duration', y='price',
		size='duration', color='class', hover_name='class', log_x=True, size_max=60)
	fig_.update_layout(width=800)
	st.write(fig_)


# artificial intelligence page
def ai_function():
	st.title('AI form for price prediction for flights')
	inputs = dict()
	with st.form("ai form"):

		airline = st.selectbox('choose airline',(set(dataset['airline'])))
		flight = st.text_input('enter flight id')
		stops = st.number_input('enter a number of stops on journey')
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

		st.text(inputs)



	











# creating sidebar menu
with st.sidebar:
	selected = option_menu(
			menu_title = 'Menu',
			options = ['Analysis', 'AI', 'About Me']
		)

if selected == 'Analysis':
	st.title('Analysis')
	analysis_function()

if selected == 'AI':
	ai_function()

if selected == 'About Me':
	st.title('About Me')


