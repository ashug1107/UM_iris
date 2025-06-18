import flask
from flask import Flask, request, render_template
from flask_wtf import FlaskForm
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained SVM model
'''RF_pkl_filename = 'model.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'rb')
model = pickle.load(RF_Model_pkl)'''
SVM_Model_pkl = pickle.load(open('SVM.pkl','rb'))
#RF_Model_pkl.close()

@app.route('/')
def home():
    return render_template('iris.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])

        data = np.array([[sl, sw, pl, pw]])
        prediction = SVM_Model_pkl.predict(data)[0]

        return render_template('iris.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')