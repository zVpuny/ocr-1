package org.ocr;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.RgbToGrayscale;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.NadamUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

import static org.ocr.SaveData.*;

public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);
    public static final String MODEL_FOLDER = System.getProperty("user.home")+"/OCR_data/model";

    private static final int[] labels =new int[]{0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 7, 8, 9};
    public static void main(String[] args) throws Exception {


        MultiLayerNetwork model = loadModel();
        model.init();

        NativeImageLoader nil = new NativeImageLoader(28,28,1);


        INDArray image=nil.asMatrix(new File("D:\\ocr-test\\test-letter-e2.png"));
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);
        INDArray output = model.output(image);
        System.out.println(output);
        System.out.println("TEST :"+ LetterMapping.getLetterOfId(labels[output.argMax().getInt(0)]));
        /*trainNewModel("model4",MODEL_FOLDER,0.0015,0.01,0.9,"XAVIER",1);*/








        //Contour.cropLetters("D:\\ocr-test\\roi-0.jpg");
        //Contour.cropWords("D:\\ocr-test\\line3.jpg");
        //String string = Contour.cropLines("D:\\ocr-test\\2.jpg",model);
        //System.out.println(string);
    }

    private static void trainAndEvaluateModel(int epochNum, File modelFolder, String modelFilename, DataSetIterator trainSet, DataSetIterator testSet, MultiLayerNetwork model) throws IOException {
        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(50), new EvaluativeListener(testSet, 1, InvocationType.EPOCH_END));

        model.fit(trainSet, epochNum);

        if(!modelFolder.exists()){
            modelFolder.mkdirs();
         }
        model.save(new File(modelFolder +"/"+modelFilename+".zip"));


        log.info("Evaluate model...");
        Evaluation eval = model.evaluate(testSet);
        System.out.println(eval.stats(false,true));
    }

    private static DataSetIterator loadTestingData(int batchSize) throws Exception {
        log.info("Load testing data...");
        if(!new File(CUSTOM_TEST_FOLDER).exists() ){
            log.info("No testing data to load. Saving testing data...");
            return SaveData.saveCustomTestData(batchSize);
        }
        else{
            DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(new File(CUSTOM_TEST_FOLDER),"custom-test-%d.bin");
            return new AsyncDataSetIterator(existingTestData);
        }
    }

    private static DataSetIterator loadTrainingData(int batchSize) throws Exception {
        log.info("Load training data...");
        if(!new File(CUSTOM_TRAIN_FOLDER).exists() ){
            log.info("No training data to load. Saving training data...");
            return SaveData.saveCustomTrainData(batchSize);
        }
        else{
            DataSetIterator existingTrainData = new ExistingMiniBatchDataSetIterator(new File(CUSTOM_TRAIN_FOLDER),"custom-train-%d.bin");
            return new AsyncDataSetIterator(existingTrainData);
        }
    }

    private static MultiLayerNetwork createModel(int seed, int nChannels, int outputNum,Double l2,Double learningRate , Double momentum, String weightInit){



        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(l2) // 0.0005
                .updater(new Nesterovs(learningRate, momentum)) // 0.01 , 0.9
                .weightInit(WeightInit.valueOf(weightInit)) // XAVIER
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();
        return new MultiLayerNetwork(conf);
    }

    public static MultiLayerNetwork loadModel() throws Exception{
        return MultiLayerNetwork.load(new File(MODEL_FOLDER+"/model4.zip"),true);
    }

    private static void xor() {
        INDArray input = Nd4j.zeros(4,2);
        INDArray knownOutput = Nd4j.zeros(4,1);

        input.putScalar(new int[]{0, 0}, 0);
        input.putScalar(new int[]{0, 1}, 0);
        input.putScalar(new int[]{1, 0}, 0);
        input.putScalar(new int[]{1, 1}, 1);
        input.putScalar(new int[]{2, 0}, 1);
        input.putScalar(new int[]{2, 1}, 0);
        input.putScalar(new int[]{3, 0}, 1);
        input.putScalar(new int[]{3, 1}, 1);

        knownOutput.putScalar(new int[]{0}, 0);
        knownOutput.putScalar(new int[]{1}, 1);
        knownOutput.putScalar(new int[]{2}, 1);
        knownOutput.putScalar(new int[]{3}, 0);

        DataSet dataSet = new DataSet(input,knownOutput);

        MultiLayerConfiguration cfg = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.UNIFORM)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .nIn(2)
                        .nOut(3)
                        .build())
                .layer(1,new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.SIGMOID)
                .nIn(3)
                .nOut(1)
                .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(cfg);
        network.init();
        network.setLearningRate(0.7);

        System.out.println(network.summary());

        for( int i=0; i < 10000; i++ ) {
            network.fit(dataSet);
        }

        INDArray output = network.output(input);
        Evaluation eval = new Evaluation();
        eval.eval(knownOutput,output);
        System.out.println(eval.stats());
        INDArray test = Nd4j.zeros(1,2);
        test.putScalar(new int[]{0,0},0);
        test.putScalar(new int[]{0,1},1);
        System.out.println("Test:"+network.output(test).getScalar(new int[]{0,0}));
    }

    public static String getLetter(Mat roi, MultiLayerNetwork model) throws IOException {

        NativeImageLoader nil = new NativeImageLoader();
        INDArray image = nil.asMatrix(roi);
        /*HighGui.imshow("aaa", roi);
        HighGui.moveWindow("aaa", 500, 0);
        HighGui.waitKey(0);
        HighGui.destroyWindow("aaa");*/
        INDArray output = model.output(image);

        return LetterMapping.getLetterOfId(labels[output.argMax().getInt(0)]);
    }


    public int getIndexOfLargest( int[] array )
    {
        if ( array == null || array.length == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }


    public static void trainNewModel(String filename, String filepath,Double l2,Double learningRate, Double momentum,String weightInit,Integer epochNum) throws Exception {
        int batchSize = 64;
        int outputNum = 62;
        int seed = 123;
        int nChannels=1;

        File modelFolder = new File(filepath);

        DataSetIterator train = loadTrainingData(batchSize);
        DataSetIterator test = loadTestingData(batchSize);

        MultiLayerNetwork model = createModel(seed,nChannels,outputNum,l2,learningRate,momentum,weightInit);
        model.init();


        trainAndEvaluateModel(epochNum, modelFolder,filename, train, test, model);
    }
}
