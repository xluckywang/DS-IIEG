'''
There are several points to note in predict.by
1. Batch prediction is not possible. If you want to perform batch prediction, you can use os. listdir() to traverse the folder and use Image.open to open the image file for prediction.
2. If you want to save the prediction results as a txt file, you can use open to open the txt file and write it using the write method. You can refer to the txt_annotation.py file.
'''
from PIL import Image

from classification import Classification

classfication = Classification()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        class_name = classfication.detect_image(image)
        print(class_name)
