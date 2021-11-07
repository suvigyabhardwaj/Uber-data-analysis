#importing essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        message = request.form['message']
        textbox = [message]
        
        # Read in the data with `read_csv()'
        data = pd.read_csv('CleanedData.csv')
        data = data.drop('Unnamed: 0', axis = 1)
        
        #-------------------------------------------------------------------------------------------------------------------
        #create a binary class which is a copy of sentiment class but without 3 star rating
        data['binary_class'] = np.where(data['ride_rating']>3,1,0)
        #---------------------------------------------------------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(data['CleanRide_review'], data['binary_class'], random_state = 0)
        #---------------------------------------------------------------------------------------------------------------------
        #creating variable which assigns X_train to numbers
        vect = CountVectorizer().fit(X_train)
        X_train_vectorized = vect.transform(X_train)
        #---------------------------------------------------------------------------------------------------------------------
        model = LogisticRegression()
        model.fit(X_train_vectorized, y_train)

        sentiment = model.predict(vect.transform(textbox))
        
#        print(str(sentiment)[1:-1])
        if(sentiment == [1]):
            return render_template('result_happy.html', prediction=sentiment)
        else:
            return render_template('result_sad.html', prediction=sentiment)
    
    

if __name__ == '__main__':
	app.run(debug=True, port=8080)
    #app.run(debug=True, port=4996)
