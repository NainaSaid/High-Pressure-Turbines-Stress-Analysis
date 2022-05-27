import numpy as np
from flask import Flask, request, render_template
import pickle
import webbrowser
from threading import Timer
import sklearn


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template('index.html',prediction_text="Stress will be {}".format(output))


# if __name__ == "__main__":
# #     app.run(debug=True)

#if __name__ == "__main__":
 #   app.run(host="0.0.0.0",port="8080",debug=True)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')


if __name__ == "__main__":
    Timer(1, open_browser).start();
    app.run(port=8080)