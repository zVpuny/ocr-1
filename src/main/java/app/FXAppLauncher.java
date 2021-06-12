package app;

import com.sun.javafx.application.LauncherImpl;

public class FXAppLauncher {
    public static void main(String[] args) {
        LauncherImpl.launchApplication(FXApp.class,AppPreloader.class,args);
    }
}
