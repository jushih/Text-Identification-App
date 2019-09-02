# Jinwen-era Chinese Text Identification App
This repo contains code to build a Flask app that utilizes deep learning to identify ancient Chinese text from images. The dataset was compiled by Ryan Harper of the Portland Data Science Group. The Chinese images can be found under the data folder labeled with their Big 5 encoding. For example, the encoding b08a represents the character 馬. Selecting only Jinwen era images, a CNN model was trained to predict the character. The model achieved an accuracy of 75% on test data and the training code can be found in the models folder. The model was then incorporated into a Python flask app where a user can submit images to be identified.

![](/src/images/appdemo.gif)
