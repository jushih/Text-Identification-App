#!/Users/julieshih/anaconda/bin/python
from src.processing.utils import * 
#import src.processing.config as cfg
from keras.models import model_from_json
from keras.models import load_model
import pickle
import os
import json
from flask import Flask, request, render_template, send_from_directory


#img_dir, metadata_dir, model_dir, search_img_dir = set_paths(cfg.PATH)
from keras import backend as K

K.clear_session()

print('Loading trained model...')


model = model_from_json(open('models/model.json').read())
model.load_weights('models/model_weights.h5')
model._make_predict_function()


app = Flask(__name__, static_folder='/Users/julieshih/workspace/Text-Identification-App/data',root_path='src/')

# define apps home page
@app.route('/')
def index():
    return render_template('index.html')

# define upload function
@app.route('/upload',methods=['POST'])
def upload():

    # delete previous uploads so the generator doesn't read them
    for file in os.listdir('src/uploads/img/'):
        os.remove('src/uploads/img/'+file)

    # upload image
    upload_dir = 'src/uploads/img/' 

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    for img in request.files.getlist('file'):
        img_name = img.filename
        destination = '/'.join([upload_dir, img_name])
        img.save(destination)

    print(destination, upload_dir)

    # generate prediction and return example characters
    big5, character_samples = classify(model, 'src/uploads/')
    character = bytes.fromhex(big5).decode('big5')

    print (character, character_samples)
    return render_template('result.html', big5=big5, character=character, image_name=img_name, result_paths=character_samples)

@app.route('/src/uploads/img/<filename>')
def send_image(filename):
    return send_from_directory('/Users/julieshih/workspace/Text-Identification-App/src/uploads/img/', filename)


if __name__ == "__main__":
    app.run(port=5000, debug=True)

