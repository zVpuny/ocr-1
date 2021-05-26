import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.ocr.App;

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
        stage.setScene(new Scene(root));
        stage.show();
    }
}
