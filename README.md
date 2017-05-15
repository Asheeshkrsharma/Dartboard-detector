# Dartboard-detector

Instructions for running the program.

Execute the following command to build the file:

1. bash buildDetection.sh DartboardDetector.cpp

Execute the following command to execute the file providing the test_image_name.jpg as parameter.

2. ./a.out <location of test_image_name.jpg>

3. The output image will be generated with "detected.jpg" as name.

In task 2 we calculated F1 score for dart5.jpg using F1_score.cpp

b.3) Measures and rules to calculate the F1-score meaningfully from ground truth.**

We decided to tag the faces (ROIs) as a method of providing ground truth. These ROIs are the positions and minimum bounding areas which inscribe the faces. For this instance, image: dart5.jpg is considered (ROIs shown in Figure 3) which has the $F_1$ score of 0.88. Once the faces are tagged, two measures are employed:

The area of detected and ground truth ROIs and The Euclidean distance b/w the centres of the detected and ground truth ROIs.

From these measures, following rules are set to calculate the F1-score.

a. If the area of intersection b/w the detected and any ground truth ROI is $\geq$ to 90% of the given ground truth ROI.

b. If the Euclidian distance b/w the centres of the two ROIs is $\leq$ a threshold of 5.

if the logical AND of the two rules is true, then the detected ROI is considered to be a True positive.

![alt text](https://github.com/Asheeshkrsharma/Dartboard-detector/blob/master/3.png "partice filter")


![alt text](https://github.com/Asheeshkrsharma/Dartboard-detector/blob/master/4.png "A* search")
