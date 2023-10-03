from flask import Flask, request, render_template, redirect
import os

from flask.helpers import url_for

app = Flask(__name__)

@app.route("/")
@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET"])
def virtual_mouse():
    if request.method == "GET":
        path_list = (os.path.dirname(__file__)).split("\\")
        print(path_list)
        if path_list[-1] == "flask_app":
            path_list.pop(-1)
            print(path_list)    
        final_path = "\\".join(path_list)
        print(final_path)
        os.chdir(final_path+"\\SLR")
        print(os.getcwd())
        print("################ Directory Changed ################")
        os.system("python predict.py")

        
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)