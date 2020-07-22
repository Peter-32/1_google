import flask
from flask import Flask, render_template, redirect, request, url_for
from jinja2 import Template
from jinja2 import Environment, PackageLoader, select_autoescape

app = Flask(__name__)

if True:
# if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

@app.route('/')
def hello_world():
    return flask.render_template('index.html', name='')

@app.route('/search', methods=['POST'])
def search():
    result = request.form
    query = result.get("query")
    return flask.render_template("search.html", query=query)

if __name__ == '__main__':
    app.run()
