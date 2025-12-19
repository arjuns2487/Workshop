from flask import Flask,render_template,jsonify,request
import pandas as pd

from sklearn import linear_model
df=pd.read_csv('./homeprices.csv')
median=df['bedrooms'].median()
df.fillna(median,inplace=True)
fea=df.drop('price',axis=1)
labels=df.price

model=linear_model.LinearRegression()
model.fit(fea,labels)

app=Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data=request.json
    area=float(data["area"])
    age=float(data["age"])
    bed=float(data["bedrooms"])
    predicted_value=model.predict([[area,bed,age]])
    return jsonify({"predicted_value":predicted_value[0]})

if __name__=="__main__":
    app.run(debug=True)
