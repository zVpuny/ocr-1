package org.ocr;

import org.opencv.core.MatOfPoint;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class LetterContour {
    MatOfPoint mainContour;
    List<MatOfPoint> childContour;

    public LetterContour(MatOfPoint mainContour, List<MatOfPoint> childContour) {
        this.mainContour = mainContour;
        if(!Objects.isNull(childContour)) {
            this.childContour = new ArrayList<>(childContour);
        }
        else {
            this.childContour = null;
        }
    }

    public LetterContour(MatOfPoint mainContour) {
        this(mainContour,null);
    }

    public MatOfPoint getMainContour() {
        return mainContour;
    }

    public void setMainContour(MatOfPoint mainContour) {
        this.mainContour = mainContour;
    }

    public List<MatOfPoint> getChildContour() {
        return childContour;
    }

    public void setChildContour(List<MatOfPoint> childContour) {
        this.childContour = childContour;
    }
}
