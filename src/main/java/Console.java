import javafx.application.Platform;
import javafx.scene.control.TextArea;

import java.io.IOException;
import java.io.OutputStream;

public class Console extends OutputStream {
    TextArea outputArea;

    public Console(TextArea outputArea) {
        this.outputArea = outputArea;
    }

    @Override
    public void write(int i) throws IOException {
        if(Platform.isFxApplicationThread()) {
            outputArea.appendText(String.valueOf((char) i));
        }
        else {
            Platform.runLater(()-> outputArea.appendText(String.valueOf((char) i)));
        }
    }
}
