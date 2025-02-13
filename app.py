from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        # Process form data
        data = {
            "Age": int(request.form['age']),
            "Sex": int(request.form['sex']),
            "ChestPainType": (request.form['chest-pain']),
            "RestingBP": (request.form['resting-bp']),
            "Cholesterol": (request.form['cholesterol']),
            "FastingBS": (request.form['fasting-bs']),
            "MaxHR": (request.form['max-hr']),
            "ExerciseA0gi0a": (request.form['exercise-angina']),
            "Oldpeak": float(request.form['oldpeak']),
            "ST_Slope": (request.form['st-slope']),
        }
        
        # Create DataFrame and predict
        df = pd.DataFrame([data])
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1] * 100

        return render_template('index.html', prediction=prediction, probability=probability)
    
    return render_template('index.html', prediction=None, probability=None)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)