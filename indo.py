from flask import Flask, render_template, redirect, url_for, request
from keras.models import load_model

model = load_model('uji6.tf')
app = Flask(__name__)

@app.route("/indo")
def index():
    return render_template('indo.html')

@app.route("/submit", methods=["POST"])
def submit():
    text = request.form.get("narasi")
    prob = model.predict([text])[0]
    label = prob.argmax()
    factor = prob.max()

    return redirect(url_for('indo', label=label, factor=factor))

# @app.route("/index")
# def indo():
#     return render_template('index.html')
