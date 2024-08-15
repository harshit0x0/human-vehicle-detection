from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO
import cv2

# Initialize the Flask application
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the YOLO model
model = YOLO("./best.pt")
model.to('cpu')

# homepage
@app.route('/') 
def index(): 
    return render_template('/index.html')  #get route

@app.route('/new') 
def index2():
    return render_template('/index3.html') #get route


@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file to the upload folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Perform inference on the image
            img = cv2.imread(filepath);
            resizedImg = cv2.resize(img,(640,640))
            results = model.predict(resizedImg,conf=0.25);

            # Save the image in output folder
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_' + file.filename)
            results[0].save(output_path);

            return render_template('upload.html', uploaded_image=file.filename, output_image='output_' + file.filename)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
