package org.ocr;

import org.datavec.image.loader.NativeImageLoader;
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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

import static org.ocr.SaveData.CUSTOM_TEST_FOLDER;
import static org.ocr.SaveData.CUSTOM_TRAIN_FOLDER;

public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);
    public static final String MODEL_FOLDER = "src/main/resources/model";

    private static final int[] labels = new int[]{0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 7, 8, 9};


    private static void trainAndEvaluateModel(int epochNum, File modelFolder, String modelFilename, DataSetIterator trainSet, DataSetIterator testSet, MultiLayerNetwork model) throws IOException {
        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(50), new EvaluativeListener(testSet, 1, InvocationType.EPOCH_END));

        model.fit(trainSet, epochNum);

        if (!modelFolder.exists()) {
            modelFolder.mkdirs();
        }
        model.save(new File(modelFolder + "/" + modelFilename + ".zip"));

        log.info("Evaluate model...");
        Evaluation eval = model.evaluate(testSet);
        System.out.println(eval.stats(false, true));
    }

    private static DataSetIterator loadTestingData(int batchSize) throws Exception {
        log.info("Load testing data...");
        if (!new File(CUSTOM_TEST_FOLDER).exists()) {
            log.info("No testing data to load. Saving testing data...");
            return SaveData.saveCustomTestData(batchSize);
        } else {
            DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(new File(CUSTOM_TEST_FOLDER), "custom-test-%d.bin");
            return new AsyncDataSetIterator(existingTestData);
        }
    }

    private static DataSetIterator loadTrainingData(int batchSize) throws Exception {
        log.info("Load training data...");
        if (!new File(CUSTOM_TRAIN_FOLDER).exists()) {
            log.info("No training data to load. Saving training data...");
            return SaveData.saveCustomTrainData(batchSize);
        } else {
            DataSetIterator existingTrainData = new ExistingMiniBatchDataSetIterator(new File(CUSTOM_TRAIN_FOLDER), "custom-train-%d.bin");
            return new AsyncDataSetIterator(existingTrainData);
        }
    }

    private static MultiLayerNetwork createModel(int seed, int nChannels, int outputNum, Double l2, Double learningRate, Double momentum, String weightInit) {
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
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();
        return new MultiLayerNetwork(conf);
    }

    public static MultiLayerNetwork loadModel() throws Exception {
        return MultiLayerNetwork.load(new File(MODEL_FOLDER + "/model4.zip"), true);
    }

    public static String getLetter(Mat roi, MultiLayerNetwork model) throws IOException {

        NativeImageLoader nil = new NativeImageLoader();
        INDArray image = nil.asMatrix(roi);
        INDArray output = model.output(image);

        return LetterMapping.getLetterOfId(labels[output.argMax().getInt(0)]);
    }

    public static void trainNewModel(String filename, String filepath, Double l2, Double learningRate, Double momentum, String weightInit, Integer epochNum) throws Exception {
        int batchSize = 64;
        int outputNum = 62;
        int seed = 123;
        int nChannels = 1;

        File modelFolder = new File(filepath);

        DataSetIterator train = loadTrainingData(batchSize);
        DataSetIterator test = loadTestingData(batchSize);

        MultiLayerNetwork model = createModel(seed, nChannels, outputNum, l2, learningRate, momentum, weightInit);
        model.init();


        trainAndEvaluateModel(epochNum, modelFolder, filename, train, test, model);
    }
}
