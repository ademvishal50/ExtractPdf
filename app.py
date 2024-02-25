from flask import Flask, render_template, request
from flask_cors import CORS
from model import extract_text_from_pdf,display_res

text = ''
app = Flask(__name__)
CORS(app)
    
@app.route("/",methods=["GET","POST"])
def index():
    
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_and_read_pdf():
    global text
    if "pdfFile" not in request.files:
        return "No PDF file uploaded", 400
    # print(type(request))
    # print(request.files["pdfFile"])
    pdf_file = request.files["pdfFile"]

    if pdf_file.filename == "":
        return "No selected file", 400

    pdf_content = pdf_file.read()
    text = extract_text_from_pdf(pdf_content)
    

    if text:
        value="Successfully Uploaded"
        return render_template("index.html",value=value)
    else:
        value = "Try to Re-Upload File"
        return render_template("index.html",value=value)

@app.route("/display", methods=["GET","POST"])
def display_sentence():
    if request.form.get('action1') == "Send":
        query = request.form['text']
        result = display_res(query,text)
        return render_template("index.html",value=result)
    

if __name__ == "__main__":
    app.run(port=5000)


