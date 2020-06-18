from flask import Flask, url_for, redirect, render_template, request
from flask_mysqldb import MySQL
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import bookrecs as br

books = br.books
index = br.indx

booklist = books.values.tolist()
authlist = books["author"].values.tolist()

app = Flask(__name__)
mysql = MySQL(app)


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        bookn = request.form["nam"]
        return redirect(url_for("recs", usr=bookn))
    else:
        return render_template("in.html")


@app.route("/<usr>")
def recs(usr):
    ind = index[usr]
    df = br.genre_recommendations(usr)
    li = df.values.tolist()
    return render_template("recs.html", bookdata=li, index=ind, books=booklist)


@app.route("/Genres", methods=["POST", "GET"])
def genres():
    return render_template("Genres.html", gen=sorted(list(set(books["genre1"]))))


@app.route("/Genres/<gnr>", methods=["POST", "GET"])
def genrebooks(gnr):
    df = br.genre_books(gnr)
    li = df.values.tolist()
    size = len(li)
    return render_template("genrebooks.html", bookdata=li, size=size, genre=gnr)


@app.route("/Authors", methods=["POST", "GET"])
def authors():
    return render_template(
        "Authors.html",
        auth=sorted(list(set([i for i in authlist if authlist.count(i) > 4]))),
    )


@app.route("/Authors/<aut>", methods=["POST", "GET"])
def authorbooks(aut):
    df = br.author_books(aut)
    li = df.values.tolist()
    size = len(li)
    return render_template("authorbooks.html", bookdata=li, size=size, auth=aut)


@app.route("/About", methods=["POST", "GET"])
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run()
