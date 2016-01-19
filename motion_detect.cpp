#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    VideoCapture cap("video.mov");
    bool update_bg_model = true;
  
    if( !cap.isOpened() )
    {
        printf("can not open video file\n");
        return -1;
    }

    namedWindow("image", WINDOW_NORMAL);
    BackgroundSubtractorMOG2 bg_model;
    int erosion_size = 5;
    int dilation_size = 5;
    int threshold_area = 7;
    int frame_no = 1;
    Mat img, fgmask, fgimg;
    Mat k_e = getStructuringElement(MORPH_RECT, Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size,erosion_size ));
    Mat k_d = getStructuringElement(MORPH_RECT, Size( 2*dilation_size + 1, 2*dilation_size+1 ), Point( dilation_size,dilation_size));
    vector<vector<Point> > contours;
    for(;;)
    {
        cap >> img;

        if( img.empty() )
            break;

        if( fgimg.empty() )
          fgimg.create(img.size(), img.type());

        bg_model(img, fgmask, update_bg_model ? -1 : 0);
        fgimg = Scalar::all(0);
        img.copyTo(fgimg, fgmask);
	//EROSION
	erode( fgmask, fgmask, k_e );
	//DILATION
        dilate( fgmask, fgmask, k_d );
	//FIND CONNECTING OBJECTS
        findContours( fgmask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	float largest = 0.0;
    	int large_obj = 0;
        vector<Moments> mu(contours.size() );
        int count = 0;

        for( count = 0; count < contours.size(); count++ )
            { 
            mu[count] = moments( contours[count], true ); 
            }
	 printf("Frame No.            : %d \n", frame_no);
         printf("Total no. of objects : %d \n", count);
         vector<Point2f> mc( contours.size() );
         for( int i = 0; i < contours.size(); i++ )
            { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
	      printf("Object%d Area: %.2f, Cetroid position: %.2f, %.2f \n", i + 1,mu[i].m00,mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
              if(mu[i].m00 > largest)
		{ largest = mu[i].m00; large_obj = i + 1;}
	      if(mu[i].m00 > threshold_area)
	         {
		 Point2i p1(mc[i].x - 30, mc[i].y);
   		 Point2i p2(mc[i].x + 30, mc[i].y);
		 Point2i p3(mc[i].x, mc[i].y - 30);
   		 Point2i p4(mc[i].x, mc[i].y + 30);

		 line(img, p1, p2, Scalar( 0, 0, 255 ), 1, 8, 0);
   		 line(img, p3, p4, Scalar( 0, 0, 255 ), 1, 8, 0);

		 circle( img,mc[i],25.0,Scalar( 0, 0, 255 ),2,1 ); 
                 }
		
        }
	printf("largest objects : object%d \n", large_obj);
	printf("-------------------------------------------------------\n");
	frame_no = frame_no + 1;
        imshow("image", img);
	
        char k = (char)waitKey(30);
        if( k == 27 ) break;
    }

    return 0;
}
