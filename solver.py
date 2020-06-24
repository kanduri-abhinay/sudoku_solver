import cv2
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

def findNextCellToFill(grid, i, j):
        for x in range(i,9):
                for y in range(j,9):
                        if grid[x][y] == 0:
                                return x,y
        for x in range(0,9):
                for y in range(0,9):
                        if grid[x][y] == 0:
                                return x,y
        return -1,-1

def isValid(grid, i, j, e):
        rowOk = all([e != grid[i][x] for x in range(9)])
        if rowOk:
                columnOk = all([e != grid[x][j] for x in range(9)])
                if columnOk:
                        # finding the top left x,y co-ordinates of the section containing the i,j cell
                        secTopX, secTopY = 3 *(i//3), 3 *(j//3) #floored quotient should be used here. 
                        for x in range(secTopX, secTopX+3):
                                for y in range(secTopY, secTopY+3):
                                        if grid[x][y] == e:
                                                return False
                        return True
        return False

def solveSudoku(grid, i=0, j=0):
        i,j = findNextCellToFill(grid, i, j)
        if i == -1:
                return True
        for e in range(1,10):
                if isValid(grid,i,j,e):
                        grid[i][j] = e
                        if solveSudoku(grid, i, j):
                                return True
                        # Undo the current cell for backtracking
                        grid[i][j] = 0
        return False


def process(img0):
    #convert the image from colour to gray
    img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    #apply the gaussianblur
    img = cv2.GaussianBlur(img,(5,5),0)
    #apply adaptivethreshold
    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,7,7)
    
    #find linear segments using HaughlinesP
    lines=cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    #img=cv2.erode(img,np.ones((2,2)),iterations=1)
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
    img1=img.copy()
    #find all contours
    contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #find the contour with the maximum area
    areas = [cv2.contourArea(c) for c in contours]
    cnt=0
    max_index=0
    pts1=[]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        print(areas[max_index])
        epsilon = 0.1 * cv2.arcLength(contours[max_index], True)
        approx = cv2.approxPolyDP(contours[max_index], epsilon, True)
        img=cv2.drawContours(img1, [approx], -1, (255, 255, 255), 2)
        img0=cv2.drawContours(img0, [approx], -1, (255, 255, 255), 2)
        img2=img.copy()
        width=252
        height=252
        pts1=approx.tolist()
        pts1 = [i[0] for i in approx]
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    '''considering the minimum size of grid box be 100000 and only process the below if we found 4 points'''    
    if(len(pts1)==4 and areas[max_index]>100000.0):
        if(pts1[0][0]>pts1[1][0]):
            pts1[0],pts1[1]=pts1[1],pts1[0]
        if(pts1[2][0]>pts1[3][0]):
            pts1[2],pts1[3]=pts1[3],pts1[2]
        pts1=np.float32(pts1)
        matrix=cv2.getPerspectiveTransform(pts1,pts2)
        bwoutput=cv2.warpPerspective(img1,matrix,(width,height))
        coloutput=cv2.warpPerspective(img0,matrix,(width,height))
        ret,thresh = cv2.threshold(bwoutput,127,255,cv2.THRESH_BINARY_INV)
        grid=[[0 for i in range(9)] for j in range(9)]
        flags=[[False for i in range(9)] for j in range(9)]
        img=cv2.erode(img,np.ones((5,5)),iterations=3)
        
        x1=0
        x2=28
        y1=0
        y2=28
        print("start")
        for i in range(9):
            for j in range(9):
                im=bwoutput[y1+3:y2-3,x1+3:x2-3]
                con,_=cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                if(len(con)==0):
                    grid[i][j]=0
                else:
                    conarea=[cv2.contourArea(ch) for ch in con]
                    if(max(conarea)<30.0):
                       grid[i][j]=0
                    else:
                        flags[i][j]=True
                        im=thresh[y1+3:y2-3,x1+3:x2-3]
                        im=cv2.resize(im,(28,28))
                        im=im.reshape(28,28,1)
                        im = np.array(im).astype('float32')/255
                        im = np.expand_dims(im, axis=0)
                        grid[i][j]=np.argmax(model.predict(im))      
                x1=x2
                x2=x1+28
            x1=0
            x2=28
            y1=y2
            y2=y1+28
        print(grid)    
        if(solveSudoku(grid)): 
            print(grid)
            x=2
            y=25
            for i in range(9):
                for j in range(9):
                    if(flags[i][j]==False):
                        cv2.putText(coloutput,str(grid[i][j]),(x,y),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,0),2,cv2.LINE_AA)
                    x=x+28
                x=2
                y=y+28
            cv2.imshow("Answer",coloutput)
        else: 
            print ("No solution exists")
            return img0
    
    return img0  

model=load_model("Digit_Recognizer.h5")


cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

while(cap.isOpened()):
    ret,frame=cap.read()
    if(ret==False):
        continue
    frame=process(frame)
    cv2.imshow("image",frame)
    if(cv2.waitKey(10)==27):
        break


cv2.destroyAllWindows()
