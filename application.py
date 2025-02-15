from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

application = Flask(__name__)

# Load pre-trained models and scaler
models = {
    "elasticNet_train_test": joblib.load("elasticNet_train_test.pkl"),
    "elasticNet_kfcv": joblib.load("elasticNet_kfcv.pkl"),
    "xgboost_regression": joblib.load("xgboost_regression.pkl")
}
scaler = joblib.load("scaler.pkl")

@application.route('/analysis')
def analysis():
    return render_template('analysis.html')

@application.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    inputs = None
    if request.method == "POST":
        input_names = [
            "body_weight",
            "experience",
            "edge",
            "wp",
            "hours_climbing_training",
            "hours_weakness",
            "hours_slab",
            "hours_power",
            "hours_boards",
            "hours_projecting",
            "hours_training",
            "motivator_num"
        ]

        # Get the model from the form
        selected_model = request.form["model"] 
        model_instance = models[selected_model]

        # Get the inputs from the form
        inputs = np.array([float(request.form[name]) for name in input_names])
        inputs_np = inputs.reshape(1, -1)
        inputs_df = pd.DataFrame(inputs_np, columns=input_names)
        
        # Compute derived attribute
        percentage_pullup = inputs_df['wp'].values[0] / inputs_df['body_weight'].values[0]
        inputs_df['percentage_pullup'] = percentage_pullup

        inputs_df = inputs_df.drop(columns=['body_weight', 'wp'])
        data_names =[
            "experience",
            "edge",
            "hours_climbing_training",
            "hours_weakness",
            "hours_slab",
            "hours_power",
            "hours_boards",
            "hours_projecting",
            "hours_training",
            "percentage_pullup",
            "motivator_num"
        ]
        inputs_df = inputs_df[data_names]

        # Scale the inputs
        inputs_scaled = scaler.transform(inputs_df)

        prediction = model_instance.predict(inputs_scaled)[0] 
        prediction = round(prediction)

    return render_template("index.html", prediction = prediction, inputs = inputs)

if __name__ == "__main__":
    application.run(debug=True)
