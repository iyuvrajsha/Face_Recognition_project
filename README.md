# Face_Recognition_project
This Project is capable of reading a face and predicting the name of that person baesd on the data provided to it.

It uses haarcascade classifier for the frontal face detection and the files could be found here: https://github.com/opencv/opencv/tree/master/data/haarcascades . 
Also, the required file for this project is already present in this very repository.

The basic Algorithm that works underneath to specify Faces is a self-implemented KNN algorithm. 

## How to use:

1. Run the https://github.com/iyuvrajsha/Face_Recognition_project/blob/main/base/DetectFace.py file.
2. When the code Runs it asks for 3 opltions ie. :
    i) To add new Data.
    ii)To predict.
    iii)To Quit.

if You want to add new data , it asks for the name and then captures the frames of the person closest to camera.
(If you want to change the no. of frames then it can be changed from line no. 65 ).

if you want it to predict, the camera window opens and a box appears on the screen on the person's face with their name tagged. 
(Also it can show names for more than 1 person in the frame if they are also present in it.)

