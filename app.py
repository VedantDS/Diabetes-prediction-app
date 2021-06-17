import numpy as np
from flask import Flask, request, render_template
import joblib


app = Flask(__name__)

model = joblib.load('model-lr.obj')



@app.route('/')
def display():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
                
def predict():
    if request.method == 'POST':
    
        try:
            age = float(request.form['age'])
            bmi = float(request.form['BMI'])
            glucose = float(request.form['glucose'])
            insulin = float(request.form['insulin'])
            bp = float(request.form['BP'])
            skin = float(request.form['skin'])

            age = (age - 28)/15
            bmi = (bmi - 32.9)/9.099
            glucose = (glucose - 115)/43
            insulin = (insulin - 90)/165
            bp = (bp - 70)/16
            skin = (skin - 29)/14

            X = np.array([[glucose, bp, skin, insulin, bmi, age]])

            pred_proba = model.predict_proba(X)[0][1]

            return render_template('result.html', pred_proba = pred_proba)
        except:
            return 'Server Down!'

    else:
        return 'Method Not Allowed!'
  



if __name__ == '__main__':
    app.run(debug=True)
