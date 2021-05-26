public class OcrDocument {
    public String image;
    public String text;

    public OcrDocument(String image) {
        this.image = image;
        this.text = "";
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }
}
