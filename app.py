from flask import Flask, render_template, redirect, url_for, request
from keras.models import load_model
from matplotlib.pyplot import text
from requests import session
from flask import session

model = {
    'inggris': load_model('uji11.tf'),
    'indonesia': load_model('uji13.tf')
}

app = Flask(__name__)

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/indonesia", methods=["GET"])
def indonesia():
    session['bahasa'] = 'indonesia'
    return render_template("indo.html")

@app.route("/inggris", methods=["GET"])
def inggris():
    session['bahasa'] = 'inggris'
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    text = request.form.get("text")
    prob = model[session['bahasa']].predict([text])[0]
    label = prob.argmax()
    factor = prob.max()

    return redirect(url_for('index', label=label, factor=factor))
