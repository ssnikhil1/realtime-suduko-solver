


# Real time suduko solver


Solves a 9*9 Sudoku puzzle under one second and overlays the solved puzzle on to the original image in real-time. This repo consists of Python code for solving sudoku puzzle using Deep Learning and OpenCV in real time(Live Camera).

Tools Used: Tensorflow 2.0.0 ,keras 2.3.1 , OpenCv,numpy


## Output

<img src="https://user-images.githubusercontent.com/53668222/113466384-42f94a80-93f0-11eb-8a93-b01af365c51e.gif" width="400">


## Steps


Step-1: Capturing the image using Webcam(real time input image)

  <img src="https://user-images.githubusercontent.com/53668222/113464137-5ea82500-93df-11eb-93bd-8d6d8e452b50.jpg" width="200">
  
Step-2: Convert input image black and white .Apply threshold to remove unwanted noise
   
   <img src="https://user-images.githubusercontent.com/53668222/113464256-2523e980-93e0-11eb-9cec-f55c5a1b95f7.jpg" width="200">

Step-3: Get corner points of largest contour having 4 corners(i.e 9*9 suduko grid)

   <img src="https://user-images.githubusercontent.com/53668222/113464291-5e5c5980-93e0-11eb-8933-89bb62016926.jpg" width="200">

Step-4: Transform the axis of image(using wrap perspective)

   <img src="https://user-images.githubusercontent.com/53668222/113464458-ae87eb80-93e1-11eb-94da-e33f8d159a80.jpg" width="200"> 

Step-5: Split the image into 81 images and recognize them using digit recognizer model

   ![digit image](https://user-images.githubusercontent.com/53668222/113464634-0410c800-93e3-11eb-858b-dec028182b13.jpg)
   
   <img src="https://user-images.githubusercontent.com/53668222/113465299-cf533f80-93e7-11eb-8af2-eb8d6c256cc1.jpg" width="200">
   
Step-6:Solve suduko using Backtracking Algorithm

Step-7: Overlaying calculated result on live video
  
<img src="https://user-images.githubusercontent.com/53668222/113465980-f06a5f00-93ec-11eb-851b-e3c5b45a75f3.jpg" width="200">

<img src="https://user-images.githubusercontent.com/53668222/113466384-42f94a80-93f0-11eb-8a93-b01af365c51e.gif" width="200">












