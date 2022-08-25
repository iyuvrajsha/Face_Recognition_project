#importing all necessary files
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

train_set = []
names = []
######### KNN CODE #####
def distance(v1, v2):
   # Eucledian
   return np.sqrt(((v1-v2)**2).sum())
def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append ([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key = lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique (labels, return_counts=True)
    #Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

#taking input of one face
def takeInput():
    cap= cv2.VideoCapture(0)
    face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    face_data= []
    file_path= "./data/"
    skip=0
    face_selection =0 
    face_name= input("Enter person's name : ")
    i=0
    while True :
        ret,frame= cap.read()
        if ret==False:
            continue
        gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray_frame,1.3,5)
        faces= sorted(faces,key= lambda f: f[2]*f[3])
        for face in faces[-1:]:
            x,y,w,h = face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),2)
            offset= 0
            face_selection= frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_selection = cv2.resize(face_selection,(100,100))
        # skip+=1
        # if skip%10== 0:
            face_data.append(face_selection)
            print(len(face_data))
            i+=1
        cv2.imshow("frames",frame)
        #cv2.imshow("face selection",face_selection)

        key_pressed= cv2.waitKey(1) & 0xFF
        if i>9:
            break   
    face_data= np.asarray(face_data,dtype=object)
    face_data= face_data.reshape((face_data.shape[0],-1))
    print(face_data.shape)
    np.save(file_path+face_name+'.npy',face_data)
    cap.release()
    cv2.destroyAllWindows()

#loading data
def loadData():
    try:
        global train_set,names
        skip= 0
        dataset_path= './data/'
        face_data=[]
        labels= []
        class_id= 0
        names= []

        for fx in os.listdir(dataset_path):
            if fx.endswith('.npy'):
                print("loaded ",fx)
                names.append(fx[:-4])
                data_item = np.load(dataset_path +fx,allow_pickle=True)
                face_data.append(data_item)
                target = class_id*np.ones((data_item.shape[0],))
                class_id+=1
                labels.append(target)
        facedata_set= np.concatenate(face_data,axis=0)
        face_labels= np.concatenate(labels,axis=0).reshape((-1,1))

        print(facedata_set.shape)
        print(face_labels.shape)

        train_set=np.concatenate((facedata_set,face_labels),axis=1)
        print(train_set.shape)
    except Exception as e:
        print(e,"Problem in loading Data , probably there isn't any data or directory to it is incorrect")

#predicting face
def predict():
    loadData()
    global train_set,names
    if len(names) == 0:
        return
    cap= cv2.VideoCapture(0)
    face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    while True :
        ret,frame= cap.read()
        if ret==False:
            continue
        gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        faces= face_cascade.detectMultiScale(gray_frame,1.3,5)

        for face in faces:
            #print(1)
            x,y,w,h = face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,152,0),2)
            offset= 0
            face_selection= frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_selection = cv2.resize(face_selection,(100,100))
            #face_selection.reshape(-1,1)
            #print(face_selection)
            out= knn(train_set,face_selection.flatten())
            pred_name= names[int(out)]
            cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,152,0),2)
            #print(2)
        cv2.imshow("Face Recognition",frame)
        key_pressed= cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break   
    cap.release()
    cv2.destroyAllWindows()

inputVal =1
while inputVal:
    print("What do You Want to do? ")
    print("1.Predict ")
    print("2.Add new Face ")
    print("3.Exit Program ")
    print("Enter 1 ,2 or 3 ")
    inputVal = int(input())
    if inputVal== 1:
        predict()
    elif inputVal== 2:
        takeInput()
    else:
        break