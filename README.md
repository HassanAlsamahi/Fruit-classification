# Fruit-classification

This is a practice code to try convolutional neural networks for the first time, I've used Fruit Images Dataset from the following repository 
<br /> https://github.com/Horea94/Fruit-Images-Dataset.git 

<br /> You can find the paper here: https://www.researchgate.net/publication/321475443_Fruit_recognition_from_images_using_deep_learning

# Results
I have used python and pytorch to train this dataset to classify fruits from images, and it shows 97% accuracy.

# Architecture
<br /> Input Layer: 3
<br /> Convolutional layer 1: 15
<br /> Maxpooling Layer: 2
<br /> Convolutional Layer 2: 30
<br /> Maxpooling Layers: 2
<br /> Convolutional Layer 3: 60
<br /> Maxpooling Layers: 2
<br /> Fully Connected Linear Layer: 500
<br /> Output Lyaer: 103 Classes

# How to use
This project was made by Google Collabaratory so to use it either to use google colab and change in the train and test path, or to run the .py file from a terminal and edit the train and test path if you to use it on your local computer, the .py file will train the model every time you run, still working on commands to load and run the model without training the it every time. 
