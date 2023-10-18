
# Importing modules
from flask import Flask as fl
from flask import render_template as rd
from flask import request
from tensorflow import keras
import pickle
import os
TF_CPP_MIN_LOG_LEVEL = 3
# Loading models and preprocessing tools
Port = int(os.environ.get('PORT', 5000))
model = keras.models.load_model("model_tanh_stack_val.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Initialising flask server
app = fl(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    print('yolo')
    if(request.method == "POST"):
        print('here')
        text = str(request.form['message'])
        print(text)
        gg = tokenizer.texts_to_sequences([text])
        padded = keras.preprocessing.sequence.pad_sequences(gg, padding='post', maxlen=500)
        model_ans = model.predict(padded)
        print("Probability:", model_ans * 100)
        if(model_ans >= 0.6):
            return rd("index.html", ans="Toxic", col="#F07470")
        else:
            return rd("index.html", ans="Not toxic", col="#78EC6C")
    else:
        return("Error bad request")


@app.route("/", methods=['GET'])
def main():
    return rd("index.html", ans="", col="")


if(__name__ == "__main__"):
    app.run(debug=True,port=Port, threaded=False)
