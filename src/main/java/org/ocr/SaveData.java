package org.ocr;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class SaveData {
    private static final Logger log = LoggerFactory.getLogger(App.class);
    public static final String CUSTOM_TRAIN_FOLDER = "src/main/resources/data/saved/train";
    public static final String CUSTOM_TEST_FOLDER = "src/main/resources/data/saved/test";
    public static final String CUSTOM_TRAIN_DATA_FOLDER = "src/main/resources/data/train";
    public static final String CUSTOM_TEST_DATA_FOLDER = "src/main/resources/data/test";



    public static DataSetIterator saveCustomTrainData(int batchSize) throws Exception{
        int height = 28;
        int width = 28;
        int channels = 1;
        int seed = 123;
        Random randomNumberGenerator = new Random(seed);
        int outputNumber = 62;

        File trainData = new File(CUSTOM_TRAIN_DATA_FOLDER);

        FileSplit trainFileSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randomNumberGenerator);

        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

        ImageRecordReader imageRecordReader = new ImageRecordReader(height,width,channels,labelGenerator);
        imageRecordReader.initialize(trainFileSplit);

        log.info("Record labels" + imageRecordReader.getLabels());

        DataSetIterator trainDataIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,1,outputNumber);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(trainDataIterator);
        trainDataIterator.setPreProcessor(scaler);

        File trainFolder = new File(CUSTOM_TRAIN_FOLDER);
        trainFolder.mkdirs();

        log.info("Saving train data to "+trainFolder.getAbsolutePath() );
        int trainDataSaved = 0;

        while (trainDataIterator.hasNext()){
            trainDataIterator.next().save(new File(trainFolder,"custom-train-"+trainDataSaved+".bin"));
            trainDataSaved++;
        }

        log.info("Finished saving train data");

        return trainDataIterator;
    }

    public static DataSetIterator saveCustomTestData(int batchSize) throws Exception{
        int height = 28;
        int width = 28;
        int channels = 1;
        int seed = 123;
        Random randomNumberGenerator = new Random(seed);
        int outputNumber = 62;

        File testData = new File(CUSTOM_TEST_DATA_FOLDER);

        FileSplit testFileSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS,randomNumberGenerator);

        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

        ImageRecordReader imageRecordReader = new ImageRecordReader(height,width,channels,labelGenerator);
        imageRecordReader.initialize(testFileSplit);


        DataSetIterator testDataIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,1,outputNumber);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(testDataIterator);
        testDataIterator.setPreProcessor(scaler);

        File testFolder = new File(CUSTOM_TEST_FOLDER);
        testFolder.mkdirs();

        log.info("Saving test data to "+testFolder.getAbsolutePath() );
        int testDataSaved = 0;

        while (testDataIterator.hasNext()){
            testDataIterator.next().save(new File(testFolder,"custom-test-"+testDataSaved+".bin"));
            testDataSaved++;
        }

        log.info("Finished saving test data");

        return testDataIterator;
    }



}
