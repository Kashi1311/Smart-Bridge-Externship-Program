from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('Disease_Prediction_model.sav', 'rb'))
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route('/details')
def pred():
    return render_template('details.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    col = ['itching','continuous_sneezing','shivering','joint_pain','stomach_pain',
           'vomiting','fatigue','weight_loss','restlessness','Lethargy',
           'high_fever','headache','dark_urine','nausea','pain_behind_the_eyes',
           'constipation','abdominal_pain','diarrhoea','mild_fever','yellowing_of_eyes',
           'malaise','phlegam','congestion','chest_pain','fast_heart_rate',
           'neck_pain','dizziness','puffy_face_and_eyes','knee_pain','muscle_weakness',
           'passage_of_gases','irritability','muscle_pain','belly_pain','abnormal_menstruation',
           'increased_appetite','Lack_of_concentration','visual_disturbances','receiving_blood_transfuson',
           'coma','history_of_alcohol_consumption','blood_in_sputum','palpitations',
           'inflammatory_nails','yellow_crust_ooze']
    if request.method == 'POST':
        inputt = [str(x) for x in request.form.values()]

        b=[0]*45
        for x in range(0,45):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        b=b.reshape(1,45)
        prediction = model.predict(b)
        prediction = prediction[0]
    return render_template('results.html', prediction_text='The probabale diagnosis says it could be')

if __name__ == "__main__":
    app.run()
