from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/post/<int:post_id>")
def even_odd(post_id):
    if(post_id % 2==0):
        return "<p>Number is EVEN!</p>"
    else:
       return "<p>Number is ODD!</p>" 
    