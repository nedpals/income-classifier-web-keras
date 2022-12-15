import io
import csv
import numpy as np
import joblib
import pandas as pd
from flask import Flask, render_template, request
from tensorflow import keras

nb_model = joblib.load('models/model_naivebayes')
dt_model = joblib.load('models/model_decisiontree')
ann_model = keras.models.load_model('models/model.h5')

feature_choices = {
    'marital-status': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                       'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
    'occupation': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                   'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
                   'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',
                   'Tech-support', 'Transport-moving'],
    'relationship': ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'],
    'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],
    'sex': ['Female', 'Male'],
    'native-country': ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
                'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala',
                'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran',
                'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua',
                'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal',
                'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago',
                'United-States', 'Vietnam', 'Yugoslavia']
}

feature_labels_indices = {
    key: {
        label: idx for idx, label in enumerate(entry)
    } for key, entry in feature_choices.items()
}

feature_labels = {
    'marital-status': 'Marital Status',
    'occupation': 'Occupation',
    'relationship': 'Relationship',
    'race': 'Race',
    'sex': 'Sex',
    'native-country': 'Country of Origin'
}

classifier_choices = {
    'decision_tree': 'Decision Tree',
    'naive_bayes': 'Naive Bayes',
    'ann': 'ANN'
}

outcomes = ['<=50K', '>50K']

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

def predict_decision_tree(inputs: pd.DataFrame):
    return dt_model.predict(inputs)[0]

def predict_naive_bayes(inputs: pd.DataFrame):
    return nb_model.predict(inputs)[0]

def predict_ann(inputs: np.ndarray):
    raw_prediction = ann_model.predict(inputs)
    return int((raw_prediction > 0.5).ravel()[0])

classifiers = {
    'decision_tree': predict_decision_tree,
    'naive_bayes': predict_naive_bayes,
    'ann': predict_ann
}

@app.route("/")
def index_page():
    return render_template('index.html',
        feature_labels=feature_labels,
        feature_choices=feature_choices,
        classifier_choices=classifier_choices
    )

@app.post("/prediction")
def predict_inputs():
    inputs = []
    prediction = 'n/a'

    input_file = request.files.get('input_file', None)
    has_input_file = input_file is not None and input_file.content_type == 'text/csv'

    if has_input_file:
        input_file = request.files.get('input_file')
        input_file.stream.seek(0)
        input_data = io.StringIO(input_file.stream.read().decode("UTF8"))
        csv_reader = csv.DictReader(input_data, delimiter=',', quotechar='"')
        for row in csv_reader:
            inputs = list(map(lambda name: [name, row[name]], feature_labels.keys()))

    else:
        data = request.form
        inputs = list(map(lambda name: [name, data.get(name)], feature_labels.keys()))

    raw_inputs = list(map(lambda val: val[1], inputs))
    inputs = list(map(lambda val: feature_labels_indices[val[0]][val[1]], inputs))

    classifier = request.form.get('classifier')
    if classifier in classifiers:
        if classifier == 'ann':
            final_input = np.expand_dims(np.array(inputs), axis=0)
        else:
            final_input = pd.DataFrame([inputs], columns=feature_labels.keys())
        
        raw_prediction = classifiers[classifier](inputs=final_input)
        prediction = outcomes[raw_prediction]
    else:
        raw_prediction = []
        prediction = 'invalid classifier'

    return {
        'inputs': raw_inputs,
        'prediction': prediction,
        'raw_prediction': int(raw_prediction),
        'classifier': classifier_choices[classifier]
    }

if __name__ == "__main__":
    app.run()
