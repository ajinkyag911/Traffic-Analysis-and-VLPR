import cv2
import time
import os
import thread
import numpy as np
from subprocess import call
from openalpr import Alpr

alpr = Alpr("eu", "oalpr.conf", "runtime_data")
alpr.set_top_n(1)

cap = cv2.VideoCapture('v3.mp4')
cap1 = cv2.VideoCapture('v3.mp4')
t=time.time()

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def knn():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()        

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print "\nerror: KNN traning was not successful\n"               # show error message
        return                                                          # and exit program
   

    imgOriginalScene  = cv2.imread("abb.png")              

    if imgOriginalScene is None:                            
        print "\nerror: image not read from file \n\n"      
        os.system("pause")                                  
        return                                            
   

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                         
        print "\nno license plates were detected\n"            
    else:                                                       # else
                
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

               
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)           
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     
            print "\nno characters were detected\n\n"       
            return                                          
       

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             

        print "\nlicense plate read from image = " + licPlate.strChars + "\n"       
        print "----------------------------------------"

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

        cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file

    # end if else

    cv2.waitKey(0)					# hold windows open until user presses a key

    return
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      
    fltFontScale = float(plateHeight) / 30.0                    
    intFontThickness = int(round(fltFontScale * 1.5))           

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        

         
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         

    if intPlateCenterY < (sceneHeight * 0.75):                                                  
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      
    else:                                                                                       
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          

            
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

def detect(threadName, delay):
	t=time.time()
	prev_pl="x"
	while(True):
	    
	    ret, frame = cap.read()
	    
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    
	    
	    cv2.imwrite('img.png',gray)
	    
	    if t+2 < time.time():
	
		results = alpr.recognize_file('img.png')
		
		
		t=t+2
		i = 0
		

		for plate in results['results']:
			i += 1
     			#print("Plate #%d" % i)
     			#print("   %12s %12s" % ("Plate", "Confidence"))
     			for candidate in plate['candidates']:
         			prefix = "-"
         			if candidate['matches_template']:
	             			prefix = "*"
				current_pl=candidate['plate']
         			if(overlap(prev_pl,current_pl)==False):
					print "Detected Plate : ",current_pl
					print "\nConfidence : ",candidate['confidence']
					print "--------------------------------------"
					prev_pl=current_pl
	    os.system("rm img.png")
	    
	    if t-time.time()>50:
		break
	
def display(threadName, delay):
	while(True):
	    ret, frame1 = cap1.read()
	    
	    edges = cv2.Canny(frame1,100,100)
	    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
	    cv2.resizeWindow('Frame', 800,450)
	    cv2.namedWindow('Edges',cv2.WINDOW_NORMAL)
	    cv2.resizeWindow('Edges', 800,450)
	    cv2.imshow('Frame',frame1)
	    
	    cv2.imshow('Edges',edges)
	    k=cv2.waitKey(30)&0xff
	    if k==27:
		break

def overlap(string1,string2):
    count = 0
    for i in range(min(len(string1), len(string2))):
        if string1[i] == string2[i]:
            count = count + 1
    if count>3:    
	return True
    else:
	return False

try:
	t=time.time()
	thread.start_new_thread(detect, ("Thread-1", 2, ) )
	thread.start_new_thread(display, ("Thread-2", 4, ) )
except:
	print "Unable to start thread."
while 1:
	pass
cap.release()
cv2.destroyAllWindows()
alpr.unload()






