import cv2
import argparse
import glob
import ctypes
from LocalBinaryPatterns import LocalBinaryPatterns
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from skimage import data ,color, exposure

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training",required=True,help="path to the training images")
ap.add_argument("-c", "--colorHistro",help="Enable color histro function ( 1 to enable )")
args = vars(ap.parse_args())

#load LBP patterns and set info
desc = LocalBinaryPatterns(30, 8) # LBP with 24 point and 8 radius
datas = []
labels = []
traininglist = []
cap = cv2.VideoCapture(0)
#Color histrogram function
cHisFunc = int(args["colorHistro"])
cdatas = []
clabels = []
#Video values
x = 100
y = 100
width = 550
height = 400
#end loading

#image resize function
def imgRez( str ):
    image = cv2.imread(str)
    r = 300.0 / image.shape[1]
    dim = (300, int(image.shape[0] * r))
    image = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)
    return image
#end function

#video resize function
def vidRez ( image ):
    frame = image
    r = 300.0 / frame.shape[1]
    dim = (300, int(frame.shape[0] * r))
    frame = cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    return frame

#color histo function
def colorHisto( image ):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()
    return hist

#Message Box for convenient use
def Mbox(title, text, style):
    ctypes.windll.user32.MessageBoxW(0, text, title, style)
#Loading Training enviroment
with open(args["training"]) as f:
    for line in f:
        line = line.replace("\n","")
        traininglist.append(line)
#Ending loading path

#Feature
print "List of possible class"
for imagePath in traininglist:
    classname = imagePath[imagePath.rfind("/") + 1:]
    print classname
    for imageload in glob.glob("." + imagePath + "/*.jpg"):#use glob to load all file
        filePatch = imageload
        #image = cv2.imread(filePatch)
        image = imgRez(filePatch)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        labels.append(classname)
        datas.append(hist)
        if(cHisFunc==1):
            chist = colorHisto(image)
            clabels.append(classname)
            cdatas.append(chist)

# Train a Linear SVM on data
model = LinearSVC(C=100.0, random_state=36)
model.fit(datas, labels)

if(cHisFunc==1):
    cmodel = LinearSVC(C=100.0, random_state=48)
    cmodel.fit(cdatas, clabels)

# Start of testing
while(True):
    ret, frame = cap.read()
    #frame = cv2.bilateralFilter(frame,9,75,75)
    #frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21) #use this with superior computer
    roi = frame[y:height, x:width]
    image = vidRez(roi)
    #image = roi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Train LBM frame
    hist = desc.describe(gray)
    hist = hist.reshape(1,-1)
    prediction = model.predict(hist)[0]
    output = prediction
    #show video
    if(cHisFunc==1):
        chist = colorHisto(image).reshape(1,-1)
        cprediction = cmodel.predict(chist)[0]
        output = cprediction
        if(prediction!=cprediction):
            output = "Negative"
    cv2.putText(frame, "prediction is " + output, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 2)
    cv2.imshow('test',image)
    cv2.rectangle(frame, (x,y), (width,height), (0,255,0), 2)
    cv2.imshow('VideoCamera',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()