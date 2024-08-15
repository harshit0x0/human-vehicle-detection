import cv2                                          #computer vision- used for image processing
import os                                           #operating system- used for commands like current directory
from ultralytics import YOLO                        #importing YOLO from the library


model_path = "./best.pt"
model = YOLO(model_path)
model.to('cpu');

#testing

image_folder = './static/uploads'    #folder where images are located

#iterating over all images
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust file extensions as needed

        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        #inference drawing
        results = model.predict(img, conf=0.09)

        #display results
        results[0].show()

        output_filename = f'output/{filename}_result.jpg'
        results[0].save(output_filename)
