from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

iris_name = ['setosa', 'versicolor', 'virginica']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_file = None
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred = model.predict(features)[0]
            prediction = iris_name[pred]
            image_file = f"images/{prediction}.jpg"
        except Exception as e:
            prediction = f"입력 오류: {e}"
            image_file = None
    return render_template('index.html', prediction=prediction, image_file=image_file)

@app.route('/index2', methods=['GET', 'POST'])
def index2():
    prediction = None
    image_file = None
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred = model.predict(features)[0]
            prediction = iris_name[pred]
            image_file = f"images/{prediction}.jpg"
        except Exception as e:
            prediction = f"입력 오류: {e}"
            image_file = None
    return render_template('index2.html', prediction=prediction, image_file=image_file)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)