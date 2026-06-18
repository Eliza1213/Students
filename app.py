from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelo y métricas
model    = joblib.load('modelo_students.pkl')
metricas = joblib.load('metricas_cv.pkl')
app.logger.debug('Modelo cargado correctamente.')

# Features que RFE seleccionó
FEATURES = metricas['features']

# Encodings alfabéticos (igual que LabelEncoder)
ENCODINGS = {
    'gender':                      {'female': 0, 'male': 1},
    'race/ethnicity':              {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4},
    'parental level of education': {'associate s degree': 0, "bachelor's degree": 1,
                                    'high school': 2, "master's degree": 3,
                                    'some college': 4, 'some high school': 5},
    'lunch':                       {'free/reduced': 0, 'standard': 1},
    'test preparation course':     {'completed': 0, 'none': 1},
}

FORM_OPTIONS = {
    'gender':                      ['female', 'male'],
    'race/ethnicity':              ['group A', 'group B', 'group C', 'group D', 'group E'],
    'parental level of education': ['associate s degree', "bachelor's degree", 'high school',
                                    "master's degree", 'some college', 'some high school'],
    'lunch':                       ['free/reduced', 'standard'],
    'test preparation course':     ['completed', 'none'],
}

NUMERIC_FIELDS = ['reading score', 'writing score']


def feat_to_key(feat):
    """Convierte nombre de feature a clave del formulario."""
    return feat.replace(' ', '_').replace('/', '_')


@app.route('/')
def home():
    return render_template('formulario.html',
                           metricas=metricas,
                           features=FEATURES,
                           form_options=FORM_OPTIONS,
                           numeric_fields=NUMERIC_FIELDS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        row = {}
        for feat in FEATURES:
            raw = request.form.get(feat_to_key(feat))
            if feat in NUMERIC_FIELDS:
                row[feat] = float(raw)
            else:
                row[feat] = ENCODINGS[feat][raw]

        data_df = pd.DataFrame([row], columns=FEATURES)
        prediction = model.predict(data_df)[0]
        app.logger.debug(f'Predicción: {prediction}')

        return jsonify({'math_score': round(float(prediction), 2)})

    except Exception as e:
        app.logger.error(f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8080)