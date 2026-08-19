from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            SEX=int(request.form.get('SEX')),
            EDUCATION=int(request.form.get('EDUCATION')),
            MARRIAGE=int(request.form.get('MARRIAGE')),
            AGE=int(request.form.get('AGE')),
            PAY_0=int(request.form.get('PAY_0')),
            PAY_2=int(request.form.get('PAY_2')),
            PAY_3=int(request.form.get('PAY_3')),
            PAY_4=int(request.form.get('PAY_4')),
            PAY_5=int(request.form.get('PAY_5')),
            PAY_6=int(request.form.get('PAY_6')),
            BILL_AMT1=float(request.form.get('BILL_AMT1')),
            BILL_AMT2=float(request.form.get('BILL_AMT2')),
            BILL_AMT3=float(request.form.get('BILL_AMT3')),
            BILL_AMT4=float(request.form.get('BILL_AMT4')),
            BILL_AMT5=float(request.form.get('BILL_AMT5')),
            BILL_AMT6=float(request.form.get('BILL_AMT6')),
            PAY_AMT1=float(request.form.get('PAY_AMT1')),
            PAY_AMT2=float(request.form.get('PAY_AMT2')),
            PAY_AMT3=float(request.form.get('PAY_AMT3')),
            PAY_AMT4=float(request.form.get('PAY_AMT4')),
            PAY_AMT5=float(request.form.get('PAY_AMT5')),
            PAY_AMT6=float(request.form.get('PAY_AMT6'))
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        preds, proba = predict_pipeline.predict(pred_df)

        default_prob = float(proba[0]) * 100
        prediction = "Will Default" if preds[0] == 1 else "Will Not Default"

        if default_prob >= 70:
            risk_level = "High Risk"
            risk_color = "high"
        elif default_prob >= 40:
            risk_level = "Moderate Risk"
            risk_color = "moderate"
        else:
            risk_level = "Low Risk"
            risk_color = "low"

        return render_template('home.html',
                               results=prediction,
                               probability=round(default_prob, 1),
                               risk_level=risk_level,
                               risk_color=risk_color,
                               limit_bal=request.form.get('LIMIT_BAL'),
                               age=request.form.get('AGE'),
                               sex=request.form.get('SEX'),
                               education=request.form.get('EDUCATION'),
                               marriage=request.form.get('MARRIAGE'),
                               PAY_0=request.form.get('PAY_0'),
                               PAY_2=request.form.get('PAY_2'),
                               PAY_3=request.form.get('PAY_3'),
                               PAY_4=request.form.get('PAY_4'),
                               PAY_5=request.form.get('PAY_5'),
                               PAY_6=request.form.get('PAY_6'),
                               BILL_AMT1=request.form.get('BILL_AMT1'),
                               BILL_AMT2=request.form.get('BILL_AMT2'),
                               BILL_AMT3=request.form.get('BILL_AMT3'),
                               BILL_AMT4=request.form.get('BILL_AMT4'),
                               BILL_AMT5=request.form.get('BILL_AMT5'),
                               BILL_AMT6=request.form.get('BILL_AMT6'),
                               PAY_AMT1=request.form.get('PAY_AMT1'),
                               PAY_AMT2=request.form.get('PAY_AMT2'),
                               PAY_AMT3=request.form.get('PAY_AMT3'),
                               PAY_AMT4=request.form.get('PAY_AMT4'),
                               PAY_AMT5=request.form.get('PAY_AMT5'),
                               PAY_AMT6=request.form.get('PAY_AMT6'))

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
