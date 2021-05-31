import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ocr.App;
import org.ocr.Contour;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;


public class Controller {
    @FXML
    public TextField filesTextField;
    public TextArea resultText;
    public ImageView srcImage;
    public Button prevDocButton;
    public Button nextDocButton;
    public TextField savePathTextField;
    public TextField customModelPathField;

    public TextArea consoleArea;
    public ChoiceBox<String> weightChoiceBox;
    public Spinner<Double> momentumField;
    public Spinner<Double> learningRateField;
    public Spinner<Double> regularizationRateField;
    public Spinner<Integer> epochsNoField;
    public TextField networkSavePathField;
    public TextField networkFilenameField;
    public HBox mainHBox;
    public HBox mainHBox2;
    public VBox paramsVBox;
    public VBox consoleVBox;
    public VBox trainButtonVBox;
    public VBox savepathButtonVBox;
    public VBox savepathVBox;
    public StackPane stackPane;
    public AnchorPane mainAnchorPane;
    public HBox bottomHBox;

    int sceneWidth;
    int sceneHeigth;
    public List<OcrDocument> documents = new ArrayList<>();

    ListIterator<OcrDocument> documentsIterator;
    private final BooleanProperty canShowPrevDocButton = new SimpleBooleanProperty(false);
    private final BooleanProperty canShowNextDocButton = new SimpleBooleanProperty(false);
    boolean wasLastClickedNextDoc = true;
    public static final String test = "test";

    @FXML
    public void initialize() {
        Console console = new Console(consoleArea);
        PrintStream ps = new PrintStream(console,true);
        System.setOut(ps);
        /*System.setErr(ps);*/


        List<String> weightInitializers = new ArrayList<>();
        weightInitializers.add("ZEROS");
        weightInitializers.add("ONES");
        weightInitializers.add("NORMAL");
        weightInitializers.add("XAVIER");
        weightInitializers.add("RELU");



        ObservableList<String> observableWeightInitializer = FXCollections.observableList(weightInitializers);

        weightChoiceBox.getItems().addAll(observableWeightInitializer);
        prevDocButton.visibleProperty().bind(canShowPrevDocButton);
        nextDocButton.visibleProperty().bind(canShowNextDocButton);
    }

    void initData(int width,int height){
        sceneWidth = width;
        sceneHeigth = height;
    }


    public void runOCR(MouseEvent mouseEvent) throws IOException {
        int i = 0;
        for (OcrDocument document : documents) {

            document.setText(Contour.cropLines(document.getImage(), FXApp.model));

            if (savePathTextField.getText() != null) {
                FileUtils.write(new File(savePathTextField.getText() + "/file-" + i + ".txt"), document.getText());
                i++;
            }
        }
        resultText.setText(documents.get(0).getText());

    }

    public void addFiles(MouseEvent mouseEvent) throws FileNotFoundException {

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Choose Files");
        List<File> files = fileChooser.showOpenMultipleDialog(new Stage());
        StringBuilder filesString = new StringBuilder();
        if (files != null) {
            documents.clear();
            for (File file : files) {
                filesString.append(file.getName());
                filesString.append("; ");
                documents.add(new OcrDocument(file.getAbsolutePath()));
            }
            filesTextField.setText(filesString.toString());
        }
        srcImage.setImage(new Image(new FileInputStream(documents.get(0).getImage())));
        documentsIterator = documents.listIterator();
        if(documentsIterator.hasNext()){
            documentsIterator.next();
            canShowNextDocButton.setValue(documentsIterator.hasNext());
        }


        
    }

    public void chooseSaveDirectory(MouseEvent mouseEvent) {
        DirectoryChooser directoryChooser = new DirectoryChooser();
        directoryChooser.setTitle("Choose save directory");
        File directory = directoryChooser.showDialog(new Stage());
        if (directory != null) {
            savePathTextField.setText(directory.getAbsolutePath());
        }
    }

    public void toggleDarkMode(MouseEvent mouseEvent) {
        Node source = (Node) mouseEvent.getSource();

        if (!source.getParent().getStylesheets().contains("css/dark-theme.css")) {
            source.getParent().getStylesheets().add("css/dark-theme.css");

        } else {
            source.getParent().getStylesheets().remove("css/dark-theme.css");
        }


    }

    public void trainNetwork(MouseEvent mouseEvent) throws Exception {
        Thread thread = new Thread(()->{
            try {
                App.trainNewModel(networkFilenameField.getText(),
                        networkSavePathField.getText(),
                        regularizationRateField.getValue(),
                         learningRateField.getValue(),
                        momentumField.getValue(),
                        weightChoiceBox.getValue(),
                        epochsNoField.getValue());
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        thread.setDaemon(true);
        thread.start();

    }


    public void chooseCustomNetwork(MouseEvent mouseEvent) throws IOException {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Choose Custom Network Model");
        File customModel = fileChooser.showOpenDialog(new Stage());
        if(customModel!=null) {
        customModelPathField.setText(customModel.getAbsolutePath());
            FXApp.model.close();
            FXApp.model = MultiLayerNetwork.load(customModel, false);
            FXApp.model.init();
        }
    }

    public void chooseNetworkSaveDirectory(MouseEvent mouseEvent) {
        DirectoryChooser directoryChooser = new DirectoryChooser();
        directoryChooser.setTitle("Choose save directory");
        File directory = directoryChooser.showDialog(new Stage());
        if (directory != null) {
            networkSavePathField.setText(directory.getAbsolutePath());
        }
    }

    public void prevDoc(MouseEvent mouseEvent) throws FileNotFoundException {
        if(wasLastClickedNextDoc){
            documentsIterator.previous();
        }
        wasLastClickedNextDoc = false;

        if(documentsIterator.hasPrevious()){
            OcrDocument prevDoc = documentsIterator.previous();
            srcImage.setImage(new Image(new FileInputStream(prevDoc.getImage())));
            resultText.setText(prevDoc.getText());
        }
        this.setCanShowDocButtons();
    }

    public void nextDoc(MouseEvent mouseEvent) throws FileNotFoundException {
        if(!wasLastClickedNextDoc){
            documentsIterator.next();
        }
        wasLastClickedNextDoc = true;

        if(documentsIterator.hasNext()){
            OcrDocument nextDoc = documentsIterator.next();
            srcImage.setImage(new Image(new FileInputStream(nextDoc.getImage())));
            resultText.setText(nextDoc.getText());
        }
        this.setCanShowDocButtons();
    }

    private void setCanShowDocButtons() {
        canShowNextDocButton.setValue(documentsIterator.hasNext());
        canShowPrevDocButton.setValue(documentsIterator.hasPrevious());
    }

    public void setSizes() {
        mainAnchorPane.setPrefWidth(sceneWidth);
        mainAnchorPane.setPrefHeight(sceneHeigth);
        srcImage.setFitHeight(sceneHeigth*0.6);
        srcImage.setFitWidth(sceneWidth*0.3906);
        mainHBox.setPrefWidth(sceneWidth);
        resultText.setPrefWidth(sceneWidth*0.3906);
        resultText.setPrefHeight(sceneHeigth*0.6);
        mainHBox2.setPrefWidth(sceneWidth);
        weightChoiceBox.setPrefWidth(sceneWidth*0.1171);
        paramsVBox.setPrefWidth(sceneWidth*0.25);
        consoleVBox.setPrefWidth(sceneWidth*0.7562);
        consoleVBox.setPrefHeight(sceneHeigth);
        consoleArea.setPrefHeight(sceneHeigth*0.6666);
        trainButtonVBox.setPrefWidth(sceneWidth*0.5977);
        savepathButtonVBox.setPrefWidth(sceneWidth*0.3031);
        savepathVBox.setPrefWidth(sceneWidth*0.3547);
        stackPane.setPrefWidth(sceneWidth);
        stackPane.setPrefHeight(sceneHeigth);
        bottomHBox.setPrefWidth(sceneWidth);
        bottomHBox.setPrefHeight(sceneHeigth*0.25);


        
    }
}
