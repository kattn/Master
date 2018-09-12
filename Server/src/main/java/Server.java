import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.util.ArrayList;


public class Server extends WebSocketServer {

    public Server(InetSocketAddress address) {
        super(address);
    }

    @Override
    public void onOpen(WebSocket conn, ClientHandshake handshake) {
        System.out.println("Client connected");
    }

    @Override
    public void onClose(WebSocket conn, int code, String reason, boolean remote) {
        if(remote){
            System.out.println("Client disconnected");
        }
    }

    @Override
    public void onMessage(WebSocket conn, String messageString) {
        GsonBuilder builder = new GsonBuilder();
        Gson gson = builder.registerTypeAdapter(Message.class, new MessageTypeAdapter()).create();
        Message message = gson.fromJson(messageString, Message.class);

        if (message.getCommand() == Command.REQUEST_FEATURES) {
            ArrayList<FeatureCollection> featureCollections =  Tools.parseGeoJSON("../va.json");

            Message response = new Message();
            response.setCommand(Command.REPSONSE_FEATURES);
            response.setContent(gson.toJson(featureCollections));

            conn.send(gson.toJson(response));
        }
    }

    @Override
    public void onMessage(WebSocket conn, ByteBuffer message) {

    }

    @Override
    public void onError(WebSocket conn, Exception ex) {
        System.out.println("Error encountered from " + conn.getRemoteSocketAddress() + ":");
        System.out.println(ex);
        ex.printStackTrace();
        conn.closeConnection(0, "error encountered");
    }

    @Override
    public void onStart() {
        System.out.println("Server started successfully");
    }

    public static void main(String[] args){
        String host = "localhost";
        int port = 8080;

        WebSocketServer server = new Server(new InetSocketAddress(host, port));
        System.out.println(server.getAddress().toString());
        server.run();
    }
}