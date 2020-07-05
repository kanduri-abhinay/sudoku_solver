# sudoku_solver
This sudoku solver can any 9x9 unsolved sudoku puzzles

[![Watch the video](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQL7RLWmTlaCU5kK1Cjz03hvXtYh9A4IdpRHQ&usqp=CAU)](https://youtu.be/-Avq4TvnIpk)


## packages used

```
openCV
numpy
keras
```
## steps involved:
### Extracting sudoku grid from the frame
```
1. Convert the image from bgr to gray using cv2.cvtColor()
2. Apply cv2.GuassianBlur()
3. Apply cv2.adaptiveThreshold() so that we will get image with blackground and white digits on it,which can be later used for finding contours.
4. Use cv2.HoughlinesP()
5. Find the contours in image and also find the countour with largest area(assuming sudoku grid to be this)
6. The obtained lines using houghlinesp maynot be fully connected so use cv2.approxPolyDP()
7. Extrat the grid using cv2.getPerspectiveTransform()
```

### Recognizing digits in grid
```
1. The empty boxes in grid can be detected using counterArea of that box by setting some threshold
2. If the box is not empty ,the digit in it can be recognized using pretrained model "Digit_Recognizer.h5"
```

### Solve sudoku
```
Use any kind of algorithm to solve the sudoku as per your comfortability
```
