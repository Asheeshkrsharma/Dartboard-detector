/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - DartboardDetector.cpp
//
/////////////////////////////////////////////////////////////////////////////

// Header inclusion

#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string.h>
#include <ctime>
#include <sstream>
#include <fstream>

#include <dirent.h>
#ifdef __MINGW32__
#include <sys/stat.h>
#endif
#include <ios>
#include <stdexcept>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

/** Function Declaration */
void detectAndDisplay(Mat frame, char ImageName[100]);
vector < Rect > merge_rectangles(vector < Rect > dart_board);
void sobel(Mat img, Mat & dx, Mat & dy, Mat & mag, Mat & dist);
bool check_intersection(Rect recta, Rect rectb);
void inc_if_inside(double * * * H, int x, int y, int height, int width, int r);
bool circleLineIntersect(float x1, float y1, float x2, float y2, float cx, float cy, float cr);
bool check_intersection(int xCoord, int yCoord, int radius, int rows, int cols, Mat & coins);
int sort_circles(vector < Point3f > fit_circles);
bool hough(Mat & img_data, Mat & dist, double threshold, int minRadius, int maxRadius, double distance, Mat & h_acc, Mat & coins);
bool perform_hough_on_roi(Mat image);
static vector<Rect> filter_hog_Detections(const vector<Rect>& found, Mat& imageData);
static vector<Rect> detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData);
static vector <double> getDescriptorVector();
static vector<Rect> run_HOG_detector(char test_img[100]);
static vector<Rect> filter_overlap(std::vector <Rect> hough_rectangles,std::vector <Rect> hog_rectangles);

/** Global variables */
string cascade_name = "HaarTrainingDart.xml";

CascadeClassifier cascade;

// Function to perform Hough transform on ROI
bool perform_hough_on_roi(Mat image) {

    Mat img_grey, img_grey1; //input mat
    Mat dx, dy, mag, dist;
    Mat dx_out, dy_out, dis_out; //final output mat
    Mat h_acc, h_out; //hough space matricies

    cvtColor(image, img_grey, COLOR_BGR2GRAY);
    GaussianBlur(img_grey, img_grey, Size(3, 3), 4, 4);

    dx.create(img_grey.rows, img_grey.cols, CV_32FC1);
    dy.create(img_grey.rows, img_grey.cols, CV_32FC1);
    mag.create(img_grey.rows, img_grey.cols, CV_32FC1);
    dist.create(img_grey.rows, img_grey.cols, CV_32FC1);

    sobel(img_grey, dx, dy, mag, dist);

    //normalize arrays with max and min values of 255 and 0
    normalize(dx, dx_out, 0, 255, NORM_MINMAX, -1, Mat());
    normalize(dy, dy_out, 0, 255, NORM_MINMAX, -1, Mat());
    normalize(dist, dis_out, 0, 255, NORM_MINMAX, -1, Mat());

    h_acc.create(mag.rows, mag.cols, CV_32FC1);

    if (hough(mag, dist, 6, 16, 100, 45, h_acc, image)) {
        return true;
    } else {
        return false;
    }

  //save images
  //imwrite( "dx.jpg", dx_out );
  //imwrite( "dy.jpg", dy_out );
  //imwrite( "mag.jpg", mag );
  //imwrite( "dist.jpg", dis_out );
  //imwrite( "h_space.jpg", h_acc);
}


//Draws the rectangle after detection
static vector<Rect> filter_hog_Detections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        //rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
    }
    return found_filtered;
}

// Detects the HOG features of dartboard
static vector<Rect> detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData) {
    vector<Rect> found;
    Size padding(Size(8, 8));
    Size winStride(Size(8, 8));
    
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding);

    found = filter_hog_Detections(found, imageData);
    
    return found;
}


// Get the descriptor vector data from descriptorvector.dat
static vector <double> getDescriptorVector(){

    vector <vector <double> > descriptor_data;
    // 1. Declare the ifstream object	
    ifstream infile( "genfiles/descriptorvector.dat" );

    // 2. Get the descriptor vector data and store it to a vector datatype
    while (infile)
    {
    	string s;
    	if (!getline( infile, s )) break;

    	istringstream ss( s );
    	vector <double> record;

    	while (ss)
    		{
      			string s;
      			if (!getline( ss, s, ' ' )) break;
      			record.push_back( (double)atof(s.c_str()) );
    		}
    
    	descriptor_data.push_back( record );
     }

    return descriptor_data[0];
}

// Runs the HOG detector on the test image
static vector<Rect> run_HOG_detector(Mat &frame) {

    HOGDescriptor hog; // Use standard parameters here
    hog.winSize = Size(64, 48); // training images size
    
    //string image = test_img; // converting the char input to string.
    // 1. Set the SVM Detector using the Descriptor Vector 
    hog.setSVMDetector(getDescriptorVector());// get the Descriptor Vector and pass it to the SVM Detector
    
    // 2. Load the image to be detected.
    //Mat testImage = imread(image, CV_LOAD_IMAGE_COLOR);   
    
    // 3. Return the detected ROIs.
    return detectTest(hog, 0.3, frame);
}


// Function used to remove the ROIs that overlap with the ROIs detected by HOG.
static vector<Rect> filter_overlap(std::vector <Rect> hough_rectangles,std::vector <Rect> hog_rectangles){

    //std::vector <Rect> combined_rectangles;
    //bool ifoverlap = true;
    for(int i=0;i<hog_rectangles.size();i++){
    	for(int j=0;j<hough_rectangles.size();j++){
        	if((hog_rectangles[i] & hough_rectangles[j]).area() > 0){
			//combined_rectangles.pushback(hog_rectangles[i]);
			hough_rectangles.erase(hough_rectangles.begin()+j);
			j--;
		}	
        }
    }

    return hough_rectangles;
}

// Function to detect and draw rectangles around the detected dartboards.
void detectAndDisplay(Mat frame, char ImageName[100]) {

    std::vector < Rect > dart_board;
    std::vector < Rect > dart_board_tmp;
    Rect tmp;
    Mat frame_gray;
    std::vector <Rect> hog_detected_roi;
    std::vector <Rect> hough_filtered_roi;
    
    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(frame_gray, dart_board, 1.1, 1.9, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

    // 3. Remove the overlapping Rectangles
    dart_board_tmp=merge_rectangles(dart_board);
    dart_board.clear();
    dart_board=dart_board_tmp;
    // 4. Perform hough circle detection on ROI
    for (int i = 0; i < dart_board.size(); i++) {

        Rect R = Rect(Point(dart_board[i].x, dart_board[i].y), Point(dart_board[i].x + dart_board[i].width, dart_board[i].y + dart_board[i].height));
        Mat ROI = frame(R);
        if (perform_hough_on_roi(ROI)) {
	    hough_filtered_roi.push_back(dart_board[i]);
        }

    }    
    // 5. Perform hog detection on the image.	
    hog_detected_roi = run_HOG_detector(frame);

    // 6. Filter the hough detected ROI using HOG detected ROI	
    hough_filtered_roi = filter_overlap(hough_filtered_roi,hog_detected_roi);
    
    // 7. Draw the rectangles around the detected ROI.
    if(hough_filtered_roi.size()>0){
	for(int i=0;i<hough_filtered_roi.size();i++){
	    rectangle(frame, Point(hough_filtered_roi[i].x, hough_filtered_roi[i].y), Point(hough_filtered_roi[i].x + hough_filtered_roi[i].width, hough_filtered_roi[i].y + hough_filtered_roi[i].height), Scalar(0, 255, 0), 2);
		
	}
    }
    if(hog_detected_roi.size()>0){
	for(int i =0;i<hog_detected_roi.size();i++){
		rectangle(frame, hog_detected_roi[i].tl(), hog_detected_roi[i].br(), Scalar(64, 255, 64), 3);
	}
    }
	
    waitKey(0);
}



// Function to merge overlapping rectangles after performing the Viola-Jones algorithm.
vector < Rect > merge_rectangles(vector < Rect > dart_board) {
    int xmin, ymin, xmax, ymax;
    for (int i = 0; i < dart_board.size(); i++) {
        for (int j = 0; j < dart_board.size(); j++) {
            //Check intersection
            if (((dart_board[i] & dart_board[j]).area() > 0) & i != j) {
                //Combine rectangles by finding co-ordinates of smallest rectangle that bounds the overlapping rectangles.
                if (dart_board[i].x < dart_board[j].x) {
                    xmin = dart_board[i].x;
                } else {
                    xmin = dart_board[j].x;
                }

                if ((dart_board[i].x + dart_board[i].width) >= (dart_board[j].x + dart_board[j].width)) {
                    xmax = dart_board[i].x + dart_board[i].width;
                } else {
                    xmax = dart_board[j].x + dart_board[j].width;
                }

                if (dart_board[i].y < dart_board[j].y) {
                    ymin = dart_board[i].y;
                } else {
                    ymin = dart_board[j].y;
                }
                if ((dart_board[i].y + dart_board[i].height) >= (dart_board[j].y + dart_board[j].height)) {
                    ymax = dart_board[i].y + dart_board[i].height;
                } else {
                    ymax = dart_board[j].y + dart_board[j].height;
                }
                

                //Remove second rectangle and update dart_board;
                dart_board[i].x = xmin;
                dart_board[i].y = ymin;
                dart_board[i].width = xmax - xmin;
                dart_board[i].height = ymax - ymin;
                dart_board.erase(dart_board.begin() + j);
                j = 0;
                //break;
            }
        }
    }
return dart_board;
}


// Function to perform Sobel Transform
void sobel(Mat img, Mat & dx, Mat & dy, Mat & mag, Mat & dist) {
    float acc_dx = 0, acc_dy = 0; //accumulators
    float k1[] = {-1,
        -2,
        -1,
        0,
        0,
        0,
        1,
        2,
        1
    }; //{-2,-4,-2,0,0,0,2,4,2};//{-1,-2,-1,0,0,0,1,2,1};    //sobel kernal dx
    float k2[] = {-1,
        0,
        1,
        -2,
        0,
        2,
        -1,
        0,
        1
    }; //{-2,0,2,-4,0,4,-2,0,2};//{-1,0,1,-2,0,2,-1,0,1};    //sobel kernal dy

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            acc_dx = acc_dy = 0;

            //apply kernel/mask
            for (int nn = -1; nn < 2; nn++) {
                for (int mm = -1; mm < 2; mm++) {
                    if (i + nn > 0 && i + nn < img.rows && j + mm > 0 && j + mm < img.cols) {
                        acc_dx += (float) img.at < uchar > (i + nn, j + mm) * k1[((mm + 1) * 3) + nn + 1];
                        acc_dy += (float) img.at < uchar > (i + nn, j + mm) * k2[((mm + 1) * 3) + nn + 1];
                    }
                }
            }
            //write final values
            dx.at < float > (i, j) = acc_dx;
            dy.at < float > (i, j) = acc_dy;
            mag.at < float > (i, j) = (sqrtf(acc_dy * acc_dy + acc_dx * acc_dx)) > 100 ? 255 : 0;
            dist.at < float > (i, j) = atan2f(acc_dy, acc_dx);
            // printf("dist : %f \n", dist.at<float>(i,j) / 3.14159265f * 180 );
        }
    }
}


void inc_if_inside(double * * * H, int x, int y, int height, int width, int r) {
    if (x > 0 && x < width && y > 0 && y < height)
        H[y][x][r]++;
}


//
bool circleLineIntersect(float x1, float y1, float x2, float y2, float cx, float cy, float cr) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float a = dx * dx + dy * dy;
    float b = 2 * (dx * (x1 - cx) + dy * (y1 - cy));
    float c = cx * cx + cy * cy;
    c += x1 * x1 + y1 * y1;
    c -= 2 * (cx * x1 + cy * y1);
    c -= cr * cr;
    float bb4ac = b * b - 4 * a * c;

    if (bb4ac < 0) {
         return false; // No collision
    } else if (bb4ac = 0) {
         return false;
    } else {
        return true; //Collision
    }

}

bool check_intersection(int xCoord, int yCoord, int radius, int rows, int cols, Mat & coins) {
    // Line intersection reference
    //http://mathworld.wolfram.com/Circle-LineIntersection.html
    //Four lines from rectange
    int check = 0;
    //line one
    int l1_x1 = 0;
    int l1_x2 = 0;
    int l1_y1 = 0;
    int l1_y2 = coins.size().height;

    if (circleLineIntersect(l1_x1, l1_y1, l1_x2, l1_y2, xCoord, yCoord, radius)) {
        check++;
    }

    //line two
    int l2_x1 = 0;
    int l2_x2 = coins.size().width;
    int l2_y1 = 0;
    int l2_y2 = 0;

    if (circleLineIntersect(l2_x1, l2_y1, l2_x2, l2_y2, xCoord, yCoord, radius)) {
        check++;
    }

    //line three
    int l3_x1 = 0;
    int l3_x2 = coins.size().width;
    int l3_y1 = coins.size().height;
    int l3_y2 = coins.size().height;

     if (circleLineIntersect(l3_x1, l3_y1, l3_x2, l3_y2, xCoord, yCoord, radius)) {
        check++;
    }

    //line four
    int l4_x1 = coins.size().width;
    int l4_x2 = coins.size().width;
    int l4_y1 = 0;
    int l4_y2 = coins.size().height;
    
    if (circleLineIntersect(l4_x1, l4_y1, l4_x2, l4_y2, xCoord, yCoord, radius)) {
        check++;
    }
    if (check < 1) {

        return true;
    } else {
        return false;
    }
}

// Function to sort circles detected inside ROI
int sort_circles(vector < Point3f > fit_circles) {
    int index;
    int rad = 0;
    for (int i = 0; i < fit_circles.size(); i++) {
        if (rad < fit_circles[i].z) {
            rad = fit_circles[i].z;
            index = i;
        }
    }
    return index;
}


// Function to perform Hough 
bool hough(Mat & img_data, Mat & dist, double threshold, int minRadius, int maxRadius, double distance, Mat & h_acc, Mat & coins) {
    int radiusRange = maxRadius - minRadius;
    int HEIGHT = img_data.rows;
    int WIDTH = img_data.cols;
    int DEPTH = radiusRange;

    double * * * H;

    // Allocate memory
    H = new double * * [HEIGHT];
    for (int i = 0; i < HEIGHT; ++i) {
        H[i] = new double * [WIDTH];

        for (int j = 0; j < WIDTH; ++j)
            H[i][j] = new double[DEPTH];
    }

    for (int y = 0; y < img_data.rows; y++) {
        for (int x = 0; x < img_data.cols; x++) {
            // printf("data point : %f\n", img_data.at<float>(y,x));
            if ((float) img_data.at < float > (y, x) > 250.0) //threshold image  
            {
                for (int r = minRadius; r < radiusRange; r++) {

                    int x0 = round(x + r * cos(dist.at < float > (y, x)));
                    int x1 = round(x - r * cos(dist.at < float > (y, x)));
                    int y0 = round(y + r * sin(dist.at < float > (y, x)));
                    int y1 = round(y - r * sin(dist.at < float > (y, x)));

                    inc_if_inside(H, x0, y0, HEIGHT, WIDTH, r);
                    inc_if_inside(H, x1, y1, HEIGHT, WIDTH, r);
                }
            }
        }
    }

    // 1. Create 2D image by summing values of the radius dimension
    for (int y0 = 0; y0 < HEIGHT; y0++) {
        for (int x0 = 0; x0 < WIDTH; x0++) {
            for (int r = minRadius; r < radiusRange; r++) {
                h_acc.at < float > (y0, x0) += H[y0][x0][r];        
            }
        }
    }

    std::vector < Point3f > bestCircles;

    // 2. Compute optimal circles
    for (int y0 = 0; y0 < HEIGHT; y0++) {
        for (int x0 = 0; x0 < WIDTH; x0++) {
            for (int r = minRadius; r < radiusRange; r++) {
                if (H[y0][x0][r] > threshold) {
                    Point3f circle(x0, y0, r);
                    int i;
                    for (i = 0; i < bestCircles.size(); i++) {
                        int xCoord = bestCircles[i].x;
                        int yCoord = bestCircles[i].y;
                        int radius = bestCircles[i].z;
                        if (abs(xCoord - x0) < distance && abs(yCoord - y0) < distance) {
                            if (H[y0][x0][r] > H[yCoord][xCoord][radius]) {
                                bestCircles.erase(bestCircles.begin() + i);
                                bestCircles.insert(bestCircles.begin(), circle);
                            }
                            break;
                        }
                    }
                    if (i == bestCircles.size()) {
                        bestCircles.insert(bestCircles.begin(), circle);
                    }
                }
            }
        }
    }

    vector < Point3f > fit_circles;
    for (int i = 0; i < bestCircles.size(); i++) {
        int xCoord = bestCircles[i].x;
        int yCoord = bestCircles[i].y;
        int radius = bestCircles[i].z;
        Point2f center(xCoord, yCoord);
        if (check_intersection(xCoord, yCoord, radius, coins.cols, coins.rows, coins)) {
            int dist = sqrt(pow((xCoord - coins.size().width / 2), 2) + pow((yCoord - coins.size().height / 2), 2));
             if (dist < 45 && radius > 16) {
                Point3f point = Point3f(xCoord, yCoord, radius);
                fit_circles.push_back(point);
            }
        }
    }
  
    if (fit_circles.size() > 0) {
        int lineThickness = 4;
        int lineType = 10;
        int shift = 0;
        int max = sort_circles(fit_circles);
        stringstream ss;
        ss << fit_circles[max].z - 1;
       

        return true;
    } else {
        return false;
    }

}

/** @function main */
int main(int argc,
    const char * * argv) {
    // 1. Read Input Image
    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    //cout << argv[1] << "\n";

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    clock_t begin = clock();

    char ImageName[100];
    strcpy(ImageName, "detected.jpg");
    //strcat(ImageName, argv[1]);

    // 3. Detect Faces and Display Result
    detectAndDisplay(frame, ImageName);

    clock_t end = clock();
    double elapsed_secs = double(end - begin);
    // 3.1 Naming the file
    cout << elapsed_secs << endl;
    // 4. Save Result Image
    imwrite(ImageName, frame);

    return 0;
}
	


