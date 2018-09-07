import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Scanner;

public class Server{

    private static int defaultPort = 8080;

    public static void main(String[] args){
        try {
            GeoJSONParser.parseGeoJSON("../va.json");

            ServerSocket server = new ServerSocket(Server.defaultPort);

            System.out.println("Server has started on 127.0.0.1:" + Server.defaultPort + ".\r\nWaiting for a connection...");

            Socket client = server.accept();

            System.out.println("A client connected.");

            InputStream in = client.getInputStream();

            OutputStream out = client.getOutputStream();

            String data = new Scanner(in, "UTF-8").useDelimiter("\\r\\n\\r\\n").next();


        } catch (IOException e) {
            System.out.print(e);
        }
    }
}