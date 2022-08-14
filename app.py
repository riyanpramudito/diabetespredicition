from flask import Flask, jsonify, request, render_template, redirect, make_response
import pickle
import pandas as pd

app =Flask(__name__)


@app.route("/")
def index():
  return render_template("Medical_Data_Record.html")

@app.route("/predict", methods = ["GET", "POST"])
def pred():
  df = pd.DataFrame(columns = ['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age'])
  if request.method == 'POST':    
    #test_file = request.form
    Name = request.form["Name[first]"]
    Pregnancies = request.form["Pregnancies"]
    Glucose = request.form["Glucose"]
    BloodPressure = request.form["BloodPressure"]
    SkinThickness = request.form["SkinThickness"]
    Insulin = request.form["Insulin"]
    BMI = request.form["BMI"]
    DiabetesPedigreeFunction = request.form["DiabetesPedigreeFunction"]
    Age = request.form["Age"]
    df = df.append({'Pregnancies': Pregnancies, 'Glucose': Glucose,'BloodPressure': BloodPressure, 'SkinThickness': SkinThickness, 'Insulin': Insulin, 'BMI': BMI,'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age}, ignore_index= True)
  #return df.to_dict()
  #return test_file
  #Name = test_file["Name[first]"]
  #result = test_file["Age"]
     
  with open('model/scaler_diabetes.pkl', 'rb') as file:
    scaler = pickle.load(file)
    df.iloc[:, :] = scaler.transform(df.iloc[:, :])
  
  with open('model/rfc_cls.pkl', 'rb') as file:
    model = pickle.load(file)
    pred = model.predict(df)
    if pred == 1 :
      pred = 'high risk'
    else:
      pred = 'low risk'
  #print(df)
  #print(result)

  with open('data_collection.txt', 'a') as file:
    file.write("%s\n" % pred[0])

  #return render_template("Report_Builder.html", result = str("%.2f" %result[0]), name = Name)
  #return render_template("Report_Builder.html", result = result , name = Name)
  return render_template("Report_Builder.html", result = pred, name = Name)
if __name__ == "__main__":
  app.run(debug = True)