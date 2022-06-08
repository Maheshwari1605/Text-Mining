
import pickle
from urllib import request

from flask import Flask, render_template, request
from pycaret.classification import *
from pycaret.nlp import *

# load the model from disk

# clf=load_model('Final_Model')
# cv=load_model('Final LDA Model')
clf = pickle.load(open('lda_pickle1', 'rb'))
cv = pickle.load(open('model_pickle', 'rb'))



app = Flask(__name__)


@app.route('/')
def home():
	return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['sentence']
		data = message
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug=True)

