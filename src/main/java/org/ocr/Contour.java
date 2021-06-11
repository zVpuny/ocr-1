package org.ocr;

import com.twelvemonkeys.lang.StringUtil;
import info.debatty.java.stringsimilarity.CharacterInsDelInterface;
import info.debatty.java.stringsimilarity.CharacterSubstitutionInterface;
import info.debatty.java.stringsimilarity.WeightedLevenshtein;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class Contour {


    public static String cropWords(Mat src,MultiLayerNetwork model, boolean correctWord) throws IOException {
        StringBuilder result = new StringBuilder();
        Size rectKernelSize = new Size();
        nu.pattern.OpenCV.loadLocally();


        Mat gray = new Mat(src.rows(),src.cols(), src.type());

        Imgproc.cvtColor(src,gray,Imgproc.COLOR_BGR2GRAY);
        Mat binary = new Mat(src.rows(), src.cols(), src.type(), new Scalar(0));
        Imgproc.threshold(gray, binary, 120,255,Imgproc.THRESH_BINARY_INV);
        //showWaitDestroy("binary line",binary);
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
        String word;
        for(MatOfPoint contour: contours){
            Rect boundingRect = Imgproc.boundingRect(contour);
            if(boundingRect.height>2){
                Rect rect = new Rect (boundingRect.x - padding, boundingRect.y - padding,boundingRect.width + (padding*2),boundingRect.height+(padding*2));
                Mat roi = src.submat(rect);
                Imgproc.resize(roi,roi,new Size(0,0),2,2,Imgproc.INTER_CUBIC);
                word = cropLetters(roi, model);
                if(correctWord && !StringUtil.isNumber(word) ){
                    word = correctWord(StringUtil.toLowerCase(word),Character.isUpperCase(word.charAt(0)));
                }
                result.append(word);
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


        //showWaitDestroy("word src",src);
        Imgproc.resize(src,src,new Size(0,0),2,2,Imgproc.INTER_CUBIC);
        //showWaitDestroy("word src x2",src);



        Mat blurred =new Mat(src.rows(),src.cols(), src.type());
        Imgproc.GaussianBlur(src,blurred,new Size(3,3),1);
        //showWaitDestroy("word blurred",blurred);
        Core.addWeighted(src,2,blurred,-1,0,blurred);
        //showWaitDestroy("word addWeighted",blurred);



        Mat gray = new Mat(src.rows(),src.cols(), src.type());
        Imgproc.cvtColor(blurred,gray,Imgproc.COLOR_BGR2GRAY);
        //showWaitDestroy("word gray",gray);

        Mat binary = new Mat(src.rows(), src.cols(), src.type(), new Scalar(0));
        Imgproc.threshold(gray, binary, 0,255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
        //Imgproc.adaptiveThreshold(gray,binary,255,Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY_INV,25,6);
        //showWaitDestroy("word binary",binary);





        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary,contours,hierarchy,Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

        List<LetterContour> letterContours = new ArrayList<>();
        List<MatOfPoint> childContours = new ArrayList();
        LetterContour letterContour;
        for(int i = 0; i<hierarchy.cols();i++){
            if(hierarchy.get(0,i)[3]==-1) {
                double childContourIndex = hierarchy.get(0, i)[2];
                if (childContourIndex != -1) {
                    for(int j = 0; j<hierarchy.cols();j++){
                        if(hierarchy.get(0,j)[3] == i){
                            childContours.add(contours.get(j));
                        }
                    }
                    letterContour = new LetterContour(contours.get(i), childContours);

                } else {
                    letterContour = new LetterContour(contours.get(i));
                }
                letterContours.add(letterContour);
                childContours.clear();
            }

        }

        int padding = 0;
        sortLetterContoursByX(letterContours);


        /*for(MatOfPoint contour: contours){
            contourToDraw.add(contour);
            Rect boundingRect = Imgproc.boundingRect(contour);
            Mat paintedLetter = new Mat(500,500,CvType.CV_8U);
            paintedLetter.setTo(new Scalar(0,0,0));
            Imgproc.drawContours(paintedLetter,contours,contours.indexOf(contour),new Scalar(255,255,255),Imgproc.FILLED);
            contourToDraw.clear();
            showWaitDestroy("Painted letter", paintedLetter);
            if(boundingRect.height>2){
                Rect rect = new Rect (boundingRect.x - padding, boundingRect.y - padding,boundingRect.width + (padding*2),boundingRect.height+(padding*2));
                roi = binary.submat(rect);
                Mat resizedRoi = new Mat();
                resizedRoi = Contour.resize(roi);
                showWaitDestroy("letter",resizedRoi);
                result.append(App.getLetter(resizedRoi, model));


            }
        }*/
        Rect boundingRect = new Rect();
        Rect boundingRectOfNext = new Rect();
        List<MatOfPoint> contoursToDraw = new ArrayList<>();
        for(int i =0; i<letterContours.size();i++){
            boundingRect = Imgproc.boundingRect(letterContours.get(i).getMainContour());
            if((i+1)<letterContours.size()){
                 boundingRectOfNext = Imgproc.boundingRect(letterContours.get(i+1).getMainContour());
            }
            Mat paintedLetter = new Mat(1500,1500,CvType.CV_8U);
            paintedLetter.setTo(new Scalar(0,0,0));
            contoursToDraw.add(letterContours.get(i).mainContour);

            if(boundingRectOfNext.x+boundingRectOfNext.width <= boundingRect.x+ boundingRect.width+2 && i+1<letterContours.size()){
                contoursToDraw.add(letterContours.get(i+1).mainContour);

                i+=1;
            }
            Imgproc.drawContours(paintedLetter,contoursToDraw,-1,new Scalar(255,255,255),Imgproc.FILLED);
            if(Objects.nonNull(letterContours.get(i).childContour)){
                contoursToDraw.clear();
                contoursToDraw.addAll(letterContours.get(i).childContour);
                Imgproc.drawContours(paintedLetter,contoursToDraw,-1,new Scalar(0,0,0),Imgproc.FILLED);
            }
            contoursToDraw.clear();
            List<MatOfPoint> paintedLetterContours = new ArrayList<>();
            Mat paintedLetterContoursHierarchy = new Mat();
            Mat paintedLetterTemp = paintedLetter.clone();
            Imgproc.dilate(paintedLetter,paintedLetterTemp,Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT,new Size(3,boundingRect.height/2)));
            Imgproc.findContours(paintedLetterTemp,paintedLetterContours,paintedLetterContoursHierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

            if(paintedLetterContours.size()>0) {
                Rect paintedLetterBoundingBox = Imgproc.boundingRect(paintedLetterContours.get(0));

                paintedLetter = paintedLetter.submat(paintedLetterBoundingBox);
            }

            if(paintedLetter.width()<450 && paintedLetter.height()<450){
                Mat resizedLetter = resize(paintedLetter);
                showWaitDestroy("letter",resizedLetter);
                result.append(App.getLetter(resizedLetter,model));
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


    public static String cropLines(String filepath, MultiLayerNetwork model, boolean correctWord) throws IOException {
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
       // showWaitDestroy("full text src ",src);
        Imgproc.warpAffine(threshed,rotated,M,threshed.size());
        //Imgproc.warpAffine(src,src,M,src.size());
        //showWaitDestroy("full text tilted",src);
        Mat horProj = new Mat();
        Core.reduce(threshed,horProj,1,Core.REDUCE_AVG); // rotated zamienione na threshed

        ArrayList<Integer>  ycoords = new ArrayList<>();
        int y=0;
        int count = 0;
        boolean isSpace = false;
        for(int i=0;i<threshed.rows();++i){   // rotated zamienione na threshed
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
        Mat srcCopy = src.clone();
        for(int i=0;i<ycoords.size()-1;++i){

            Imgproc.line(srcCopy,new Point(0,ycoords.get(i)),new Point(src.cols(),ycoords.get(i)),new Scalar(0,255,0));
        }
        //showWaitDestroy("src with lines",src);
        int shift = 0;
        for(int i=0;i<ycoords.size()-1;++i){
            Rect rect = new Rect(0,ycoords.get(i),src.cols(),ycoords.get(i+1)-ycoords.get(i)-shift);

            Mat roi = src.submat(rect);

            result.append(cropWords(roi, model , correctWord));
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

    private static void sortLetterContoursByX(List<LetterContour> letterContourList){
        //sort by x coordinates of mainContour
        Collections.sort(letterContourList, new Comparator<LetterContour>() {
            @Override
            public int compare(LetterContour l1, LetterContour l2) {
                Rect rect1 = Imgproc.boundingRect(l1.mainContour);
                Rect rect2 = Imgproc.boundingRect(l2.mainContour);
                int result = 0;
                result= Double.compare(rect1.tl().x,rect2.tl().x);
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

    private static String correctWord(String wordToCorrect,boolean capitalizeFirst) throws IOException {
        WeightedLevenshtein weightedLevenshtein = new WeightedLevenshtein(
                new CharacterSubstitutionInterface() {
                    @Override
                    public double cost(char c, char c1) {
                        if (c == 'y' && c1 == 'j') return 0.1;
                        if (c == 'h' && c1 == 'i') return 0.1;
                        if (c == 'y' && c1 == 'v') return 0.1;
                        if (c == '8' && c1 == 'a') return 0.1;
                        if (c == 'g' && c1 == 'c') return 0.1;
                        if (c == 't' && c1 == 'r') return 0.1;
                        if (c == 'h' && c1 == 'u') return 0.1;
                        if (c == 'h' && c1 == 'l') return 0.1;
                        if (c == 'm' && c1 == 'u') return 0.1;
                        if (c == 'h' && c1 == '1') return 0.1;
                        if (c == 'm' && c1 == 'n') return 0.1;
                        return 1;
                    }
                },
                new CharacterInsDelInterface() {
                    @Override
                    public double deletionCost(char c) {
                        return 2;
                    }

                    @Override
                    public double insertionCost(char c) {
                        return 2;
                    }
                }
        );

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("D:\\ocr-test\\dictionary.txt"), StandardCharsets.UTF_8));
        String word;
        String closestWord = "";
        double currentDistance;
        double closestDistance = -1;
        while ((word = br.readLine()) != null ){
            currentDistance = weightedLevenshtein.distance(wordToCorrect,word);
            if(closestDistance == -1 || currentDistance<closestDistance ){
                closestDistance= currentDistance;
                closestWord = word;
            }
            if(closestDistance == 0) break;
        }
        if(capitalizeFirst) closestWord = closestWord.substring(0,1).toUpperCase() + closestWord.substring(1);
        return  closestWord;
    }

}

