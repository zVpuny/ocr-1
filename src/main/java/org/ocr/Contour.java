package org.ocr;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.*;

public class Contour {


    public static String cropWords(Mat src,MultiLayerNetwork model) throws IOException {
        StringBuilder result = new StringBuilder();
        Size rectKernelSize = new Size();
        nu.pattern.OpenCV.loadLocally();


        Mat gray = new Mat(src.rows(),src.cols(), src.type());
        Core.copyMakeBorder(src,src,12,12,12,12,Core.BORDER_CONSTANT,new Scalar(255,255,255));
        Imgproc.cvtColor(src,gray,Imgproc.COLOR_BGR2GRAY);
        Mat binary = new Mat(src.rows(), src.cols(), src.type(), new Scalar(0));
        Imgproc.threshold(gray, binary, 150,255,Imgproc.THRESH_BINARY_INV);
        showWaitDestroy("binary line",binary);
        Mat horProj = new Mat();
        Core.reduce(binary,horProj,1,Core.REDUCE_AVG);
        int textHeight=0;
        for(int i=0;i<src.rows();++i){
            if(horProj.get(i,0)[0]>0) textHeight++;
        }
        rectKernelSize.height=textHeight;
        rectKernelSize.width=textHeight*0.3;


        Mat rectKernel = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT,rectKernelSize);
        Mat dilated = new Mat();
        Imgproc.dilate(binary,dilated,rectKernel);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dilated,contours,hierarchy,Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        sortContoursByX(contours);



        int padding = 0;

        for(MatOfPoint contour: contours){
            Rect boundingRect = Imgproc.boundingRect(contour);
            if(boundingRect.height>2){
                Rect rect = new Rect (boundingRect.x - padding, boundingRect.y - padding,boundingRect.width + (padding*2),boundingRect.height+(padding*2));
                Mat roi = src.submat(rect);
                Imgproc.resize(roi,roi,new Size(0,0),2,2,Imgproc.INTER_CUBIC);
                result.append(cropLetters(roi, model));
                result.append(" ");

            }
        }
        Scalar color = new Scalar(0,0,255);
        Imgproc.drawContours(src,contours,-1,color,1,Imgproc.LINE_8,hierarchy,
                2,new Point());
        return result.toString();
    }

    public static String cropLetters(Mat src, MultiLayerNetwork model) throws IOException {
        StringBuilder result = new StringBuilder();
        nu.pattern.OpenCV.loadLocally();
        Mat roi = new Mat();


        showWaitDestroy("word src",src);
        Imgproc.resize(src,src,new Size(0,0),2,2,Imgproc.INTER_CUBIC);
        showWaitDestroy("word src x2",src);



        Mat blurred =new Mat(src.rows(),src.cols(), src.type());
        Imgproc.GaussianBlur(src,blurred,new Size(3,3),1);
        Core.addWeighted(src,2,blurred,-1,0,blurred);
        showWaitDestroy("word blurred",blurred);



        Mat gray = new Mat(src.rows(),src.cols(), src.type());
        Imgproc.cvtColor(blurred,gray,Imgproc.COLOR_BGR2GRAY);
        showWaitDestroy("word gray",gray);

        Mat binary = new Mat(src.rows(), src.cols(), src.type(), new Scalar(0));
        Imgproc.threshold(gray, binary, 170,255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
        showWaitDestroy("word binary",binary);





        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary,contours,hierarchy,Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        int padding = 0;
        for(MatOfPoint contour: contours){
            Rect boundingRect = Imgproc.boundingRect(contour);
            if(boundingRect.height>2){
                Rect rect = new Rect (boundingRect.x - padding, boundingRect.y - padding,boundingRect.width + (padding*2),boundingRect.height+(padding*2));
                roi = binary.submat(rect);
                Mat resizedRoi = new Mat();
                resizedRoi = Contour.resize(roi);
                result.append(App.getLetter(resizedRoi, model));


            }
        }
        Scalar color = new Scalar(0,0,255);
        Imgproc.drawContours(src,contours,-1,color,1,Imgproc.LINE_8,hierarchy,
                2,new Point());

        src.release();
        gray.release();
        binary.release();
        roi.release();

        return result.toString();

    }


    public static String cropLines(String filepath, MultiLayerNetwork model) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        StringBuilder result = new StringBuilder();
        Mat src = Imgcodecs.imread(filepath);
        Mat gray = new Mat();
        Imgproc.cvtColor(src,gray,Imgproc.COLOR_BGR2GRAY);
        Mat threshed = new Mat();
        Imgproc.threshold(gray,threshed,127,255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);

        Mat points = Mat.zeros(threshed.size(),threshed.type());
        Core.findNonZero(threshed,points);
        MatOfPoint mpoints = new MatOfPoint(points);
        MatOfPoint2f point2f = new MatOfPoint2f(mpoints.toArray());

        RotatedRect box = Imgproc.minAreaRect(point2f);

        Point[] vertices = new Point[4];
        box.points(vertices);


        Mat rotated = new Mat();
        Mat M  = Imgproc.getRotationMatrix2D(box.center,box.angle,1.0);
        showWaitDestroy("full text src ",src);
        Imgproc.warpAffine(threshed,rotated,M,threshed.size());
        Imgproc.warpAffine(src,src,M,src.size());
        showWaitDestroy("full text tilted",src);
        Mat horProj = new Mat();
        Core.reduce(rotated,horProj,1,Core.REDUCE_AVG);

        ArrayList<Integer>  ycoords = new ArrayList<>();
        int y=0;
        int count = 0;
        boolean isSpace = false;
        for(int i=0;i<rotated.rows();++i){
            if(!isSpace){
                if(horProj.get(i,0)[0]==0){
                    isSpace=true;
                    count=1;
                    y=i;
                }
            }
            else {
                if(horProj.get(i,0)[0]>0){
                    isSpace=false;
                    ycoords.add(y/count);
                }
                else {
                    y+=i;
                    count++;
                }
            }
        }

        ycoords.add(src.rows()-1);
        for(int i=0;i<ycoords.size()-1;++i){
            Rect rect = new Rect(0,ycoords.get(i),src.cols(),ycoords.get(i+1)-ycoords.get(i));
            Mat roi = src.submat(rect);
            result.append(cropWords(roi, model));
            result.append(System.lineSeparator());
        }

        return result.toString();

    }

    private static void showWaitDestroy(String winname, Mat img) {
        /*HighGui.imshow(winname, img);
        HighGui.moveWindow(winname, 500, 0);
        HighGui.waitKey(0);
        HighGui.destroyWindow(winname);*/
    }

    private static void sortContoursByX(List<MatOfPoint> contourList){
        //sort by x coordinates
        Collections.sort(contourList, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                Rect rect1 = Imgproc.boundingRect(o1);
                Rect rect2 = Imgproc.boundingRect(o2);
                int result = 0;
                    result = Double.compare(rect1.tl().x, rect2.tl().x);
                return result;
            }
        });
    }

    private static Mat resize(Mat image){
        int difference;
        if(image.width()>image.height()){
            difference=image.width()-image.height();
            Core.copyMakeBorder(image,image,difference/2,difference/2,0,0,Core.BORDER_CONSTANT,new Scalar(0,0,0));
        }
        else {
            difference=image.height()-image.width();
            Core.copyMakeBorder(image,image,0,0,difference/2,difference/2,Core.BORDER_CONSTANT,new Scalar(0,0,0));
        }
        Core.copyMakeBorder(image,image,(int)(0.1*image.width()),(int)(0.1*image.width()),(int)(0.1*image.width()),(int)(0.1*image.width()),Core.BORDER_CONSTANT,new Scalar(0,0,0));
        if(image.width()>28){
            Imgproc.resize(image,image,new Size(28,28),0,0,Imgproc.INTER_AREA);
        }
        else {
            Imgproc.resize(image,image,new Size(28,28),0,0,Imgproc.INTER_CUBIC);
        }


        return image;
    }

}

