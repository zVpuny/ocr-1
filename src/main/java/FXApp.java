import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.transform.Scale;
import javafx.stage.Screen;
import javafx.stage.Stage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ocr.App;

import java.awt.*;

public class FXApp extends Application {
    public static MultiLayerNetwork model;

    public static void main(String[] args) {
        launch(args);
    }

    public void init() throws Exception {
        model = App.loadModel();
        model.init();
    }

    @Override
    public void start(Stage stage) throws Exception {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/ocr-gui.fxml"));
        Parent root = loader.load();
        stage.setResizable(false);
        stage.setTitle("OCR");

        int screenWidth = (int) Screen.getPrimary().getBounds().getWidth();
        int screenHeight = (int) Screen.getPrimary().getBounds().getHeight();

        // Responsive Design
        int sceneWidth = 0;
        int sceneHeight = 0;
        if (screenWidth <= 800 && screenHeight <= 600) {
            sceneWidth = 600;
            sceneHeight = 400;
        } else if (screenWidth <= 1600 && screenHeight <= 900) {
            sceneWidth = 900;
            sceneHeight = 650;
        } else if (screenWidth <= 1920 && screenHeight <= 1080) {
            sceneWidth = 1280;
            sceneHeight = 900;
        }
        Controller controller = loader.getController();
        controller.initData(sceneWidth, sceneHeight);
        controller.setSizes();
        stage.setScene(new Scene(root, sceneWidth, sceneHeight));
        stage.show();
    }
}
