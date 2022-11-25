import io
from flask import Flask, render_template, request
import joblib
import csv
import pandas as pd

nb_model = joblib.load('models/model_naivebayes')
dt_model = joblib.load('models/model_decisiontree')

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
    'naive_bayes': 'Naive Bayes'
}

outcomes = ['<=50K', '>50K']

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/")
def index_page():
    return render_template('index.html',
        feature_labels=feature_labels,
        feature_choices=feature_choices,
        classifier_choices=classifier_choices
    )

@app.post("/response")
def response_page():
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
    df_inputs = pd.DataFrame([inputs], columns=feature_labels.keys())

    classifier = request.form.get('classifier')
    if classifier == 'decision_tree':
        prediction = outcomes[dt_model.predict(df_inputs)[0]]
    elif classifier == 'naive_bayes':
        prediction = outcomes[nb_model.predict(df_inputs)[0]]
    else:
        prediction = 'invalid classifier'

    return render_template('response.html', 
        inputs=raw_inputs,
        classifier=classifier_choices[classifier],
        prediction=prediction
    )

if __name__ == "__main__":
    app.run()