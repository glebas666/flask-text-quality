from flask import Flask, render_template, request
from txt_quality import evaluate_russian_full

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip():
            result = evaluate_russian_full(text)

    return render_template("index.html", text=text, result=result)

if __name__ == "__main__":
    app.run()
