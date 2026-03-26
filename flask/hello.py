from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

# @app.route('/')
# @app.route('/index')
# @app.route('/home2')
# def main():
#     return '<h1>Welcome to the homepage!</h1>'

# @app.route('/submit', methods=['POST'])
# def submit():
#     # return 'Form submitted successfully!'
#     return redirect("https://www.naver.com")

@app.route('/user/<username>')
def show_user(username):
    return render_template('hello.html', name=username)

@app.route('/greet')
def greet():
    name = request.args.get('name', 'Guest')
    return f'Hello, {name}!'

# @app.route('/post/<int:post_id>')
# def show_post(post_id):
#     return f'Post ID: {post_id}'

# @app.route('/api/data')
# def get_data():
#     data = {'key': 'value', 'number': 42}
#     return jsonify(data)

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['user_input']
    return f'입력된 값: {user_input}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)