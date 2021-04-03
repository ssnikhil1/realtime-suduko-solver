# Real time suduko solver


Solves a 9*9 Sudoku puzzle under one second and overlays the solved puzzle on to the original image in real-time. This repo consists of Python code for solving sudoku puzzle using Deep Learning and OpenCV in real time(Live Camera).

Tools Used: Tensorflow 2.0.0 ,keras 2.3.1 , OpenCv,numpy


### Steps


Step-1: Capturing the image using Webcam(real time input image)

  ![input_image](https://user-images.githubusercontent.com/53668222/113464137-5ea82500-93df-11eb-93bd-8d6d8e452b50.jpg)
  
Step-2: Convert input image black and white .Apply threshold to remove unwanted noise
   
   ![thresholded_image](https://user-images.githubusercontent.com/53668222/113464256-2523e980-93e0-11eb-9cec-f55c5a1b95f7.jpg)

Step-3: Get corner points of largest contour having 4 corners(i.e 9*9 suduko grid)

   ![contour_image](https://user-images.githubusercontent.com/53668222/113464291-5e5c5980-93e0-11eb-8933-89bb62016926.jpg)

Step-4: Transform the axis of image(using wrap perspective)

   ![bandicam 2021-03-27 06-19-10-699](https://user-images.githubusercontent.com/53668222/113464458-ae87eb80-93e1-11eb-94da-e33f8d159a80.jpg)

Step-5: Split the image into 81 images and recognize them using digit recognizer model

   ![digit image](https://user-images.githubusercontent.com/53668222/113464634-0410c800-93e3-11eb-858b-dec028182b13.jpg)
   
   <img src="https://user-images.githubusercontent.com/53668222/113465299-cf533f80-93e7-11eb-8af2-eb8d6c256cc1.jpg" width="200">
   
Step-6:Solve suduko using Backtracking Algorithm

Step-7: Overlaying calculated result on live video
  
   ![Uploading Hnet-image (3).gif…]()




