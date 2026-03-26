from flask import Flask, render_template, request
from train_model import predict_dog, draw_graph
import os

app = Flask(__name__)

PLOT_PATH = os.path.join('static', 'dog_plot.png')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    graph_file = None
    error = None

    if request.method == 'POST':
        try:
            length = float(request.form['length'])
            height = float(request.form['height'])

            prediction, _ = predict_dog(length, height)
            draw_graph(length, height, save_path=PLOT_PATH)
            graph_file = 'dog_plot.png'

        except Exception as e:
            error = f'입력 오류: {e}'

    return render_template(
        'index.html',
        prediction=prediction,
        graph_file=graph_file,
        error=error
    )

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)