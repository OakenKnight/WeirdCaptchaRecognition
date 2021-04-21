# Captcha detection (OCR)

### Description
Reading text from easy and relatively hard surfaces is simple project made for Soft computing course.
Recognition and reading from images using computer vision techniques 
is performed through the OpenCV library and machine learning models through Keras and Scikit-Learn
libraries, as well as the Python programming language.

### How to use project
To run the solution on your machine and check how accurate it is, you need to do the following:
1. Methods are implemented in process.py.
2. To start program, run main.py. Running the main.py file will generate a result.csv file by calling the previously implemented method for all instances in the dataset.
3. To evaluate accuracy run the evaluate.py file. This file will load the result.csv previously generated and calculate the accuracy. The output of this file is just a number that shows the percentage of accuracy of the current solution.

### Libraries

As part of this project, the following libraries are being used with Python 3.6:
* numpy
* openCV version 3.x.y
* matplotlib
* scikit-learn
* keras version 2.1.5
    * for FeedForward fully connected NN (Conv layers weren't used)
* fuzzywuzzy