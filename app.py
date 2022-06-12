from flask import Flask

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    return "Starting ML Project"


if__name__=='__main__':
    app.run(debug=True)