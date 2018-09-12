import com.google.gson.*;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;


public class Tools {

    public static ArrayList<FeatureCollection> parseGeoJSON(String path) {
        GsonBuilder builder = new GsonBuilder();
        Gson gson = builder.registerTypeAdapter(Feature.class, new FeatureTypeAdapter()).create();

        ArrayList<FeatureCollection> featureObjects = new ArrayList<>();
        try {
            FileReader reader = new FileReader(path);
            JsonArray featureCollections = gson.fromJson(reader, JsonArray.class);
            for(JsonElement ele : featureCollections){
                try {
                    featureObjects.add(gson.fromJson(ele, FeatureCollection.class));
                } catch (JsonSyntaxException e) {
                    System.out.println("Element:");
                    System.out.println(ele);
                    throw e;
                }
            }
        } catch (FileNotFoundException e) {
            System.out.println(e);
        }

        return featureObjects;
    }
}