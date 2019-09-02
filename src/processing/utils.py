#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pandas as pd
import numpy as np
import os
import pdb
from keras_preprocessing.image import ImageDataGenerator


labels = {'a440': 0,
 'a447': 1,
 'a448': 2,
 'a44a': 3,
 'a451': 4,
 'a453': 5,
 'a454': 6,
 'a455': 7,
 'a457': 8,
 'a45d': 9,
 'a45f': 10,
 'a464': 11,
 'a467': 12,
 'a46a': 13,
 'a46b': 14,
 'a46c': 15,
 'a470': 16,
 'a473': 17,
 'a474': 18,
 'a475': 19,
 'a476': 20,
 'a4a3': 21,
 'a4a4': 22,
 'a4a7': 23,
 'a4bd': 24,
 'a4c0': 25,
 'a4ce': 26,
 'a4d1': 27,
 'a4d2': 28,
 'a4e2': 29,
 'a4e5': 30,
 'a4e8': 31,
 'a4e9': 32,
 'a4eb': 33,
 'a4ec': 34,
 'a4f1': 35,
 'a4f4': 36,
 'a4fb': 37,
 'a4fd': 38,
 'a540': 39,
 'a548': 40,
 'a558': 41,
 'a55b': 42,
 'a569': 43,
 'a56b': 44,
 'a57e': 45,
 'a5a6': 46,
 'a5aa': 47,
 'a5bf': 48,
 'a5c1': 49,
 'a5ce': 50,
 'a5d0': 51,
 'a5d5': 52,
 'a5db': 53,
 'a5df': 54,
 'a5fd': 55,
 'a650': 56,
 'a656': 57,
 'a658': 58,
 'a668': 59,
 'a67e': 60,
 'a6a8': 61,
 'a6b3': 62,
 'a6b8': 63,
 'a6b9': 64,
 'a6ca': 65,
 'a6d1': 66,
 'a6d3': 67,
 'a6d5': 68,
 'a6db': 69,
 'a6e6': 70,
 'a6e7': 71,
 'a6e8': 72,
 'a751': 73,
 'a7da': 74,
 'a86b': 75,
 'a8a3': 76,
 'a8a7': 77,
 'a8ad': 78,
 'a8ae': 79,
 'a8bd': 80,
 'a8c3': 81,
 'a8c6': 82,
 'a8d3': 83,
 'a8e0': 84,
 'a8e2': 85,
 'a8e4': 86,
 'a977': 87,
 'a9ce': 88,
 'a9d2': 89,
 'a9fa': 90,
 'aa46': 91,
 'aa47': 92,
 'aacc': 93,
 'aaf7': 94,
 'aaf8': 95,
 'aaf9': 96,
 'ab42': 97,
 'ac46': 98,
 'acdb': 99,
 'ad78': 100,
 'adab': 101,
 'adfb': 102,
 'ae61': 103,
 'afe0': 104,
 'b0a8': 105,
 'b0aa': 106,
 'b0ea': 107,
 'b16f': 108,
 'b1d0': 109,
 'b3a3': 110,
 'b3bd': 111,
 'b3be': 112,
 'b54c': 113,
 'b6c0': 114,
 'b773': 115,
 'b944': 116,
 'bb44': 117,
 'bb50': 118,
 'bec7': 119}

# crawl folder for images
def get_folders(path):
  f = []
  for item in os.listdir(path):
      if os.path.join(path, item):
          f.append(path+item)
          
  return f
      
def get_files(path):
  f = []
  for root, _, filenames in os.walk(path):
    for filename in filenames:
      f.append(os.path.join(root, filename))
      
  return f

# given a character, return sample images of that character
def sample_characters(character):

    path = '/Users/julieshih/workspace/Text-Identification-App/data'
    folders = get_folders(path)
    files = get_files(path) 
    # save paths to csv and load csv instead of crawling each time
    df = pd.DataFrame({'files':files})

    # parse script
    df['script'] = np.where(df.files.str.split('/').str[-1].str.split('-').str[1].str.split('_').str[0].isna() == True, "modern",
    df.files.str.split('/').str[-1].str.split('-').str[1].str.split('_').str[0])
    # parse labels
    df['labels'] = df.files.str.split('/').str[-1].str[0:4]
    # get the relative path of files
    df['path'] = df.files.str.split('/data/').str[1]

    # keep only jinwen
    df_sample = df[df.script == 'jinwen']
    print(df_sample)

    samples = []
    # retrieve some sample images of a character
    for i in range(1, 7):
        try:
            filename = df_sample[df_sample.labels==character].path.tolist()[i]
            samples.append(filename)
        except:
            print('character does not match existing files')
    print(samples)
    return samples

def classify(model, upload_img_dir, labels=labels):

    # load search img and rescale, this generator wil contain only the search img
    datagen=ImageDataGenerator(rescale=1./255.)

    upload_generator=datagen.flow_from_directory(
        directory=upload_img_dir,
        batch_size=1,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(50,50))

    print(upload_generator)

    # make prediction
    predict_img = model.predict_generator(upload_generator)

    predicted_class_indices=np.argmax(predict_img,axis=1)

    labels = dict((v,k) for k,v in labels.items())
    prediction = [labels[k] for k in predicted_class_indices]

    print('predicted')
    # create df of results
    filenames=upload_generator.filenames
    results=pd.DataFrame({"filename":filenames,
                      "prediction":prediction})


    print(results)
    character = results.prediction[0]
    character_samples = sample_characters(character)
    return(character, character_samples)

