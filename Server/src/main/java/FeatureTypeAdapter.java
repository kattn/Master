import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;


public class FeatureTypeAdapter extends TypeAdapter<Feature> {
    @Override
    public Feature read(JsonReader reader) throws IOException {
        // the first token is the start object
        JsonToken token = reader.peek();
        Feature feature = new Feature();
        if (token.equals(JsonToken.BEGIN_OBJECT)) {
            reader.beginObject();
            while (!reader.peek().equals(JsonToken.END_OBJECT)) {
                if (reader.peek().equals(JsonToken.NAME)) {
                    String nextName = reader.nextName();

                    // Type is irrelevant
                    if (nextName.equals("type")) {
                        reader.skipValue();
                    }

                    // Identify which type of geometry
                    if (nextName.equals("geometry")) {
                        reader.beginObject();
                        while (!reader.peek().equals(JsonToken.END_OBJECT)) {
                            nextName = reader.nextName();

                            if(nextName.equals("type")){
                                String type = reader.nextString();

                                if (type.equals("Point"))
                                    feature.geometry = new Point();
                                else if (type.equals("LineString"))
                                    feature.geometry = new LineString();
                                else //Should throw a better exception
                                    throw new IOException("Met unhandled geometry type");
                            }

                            if(nextName.equals("coordinates")) {
                                token = reader.peek();
                                if (token.equals(JsonToken.BEGIN_ARRAY)) {
                                    reader.beginArray();
                                    if (feature.geometry.getClass() == Point.class) {
                                        ((Point) feature.geometry).coordinates = new ArrayList<>();
                                        while (!reader.peek().equals(JsonToken.END_ARRAY)) {
                                            ((Point) feature.geometry).coordinates.add(reader.nextDouble());
                                        }
                                    } else if (feature.geometry.getClass() == LineString.class) {
                                        ((LineString) feature.geometry).coordinates = new ArrayList<>();
                                        while (!reader.peek().equals(JsonToken.END_ARRAY)) {
                                            reader.beginArray();
                                            ArrayList<Double> coords = new ArrayList<>();
                                            while (!reader.peek().equals(JsonToken.END_ARRAY)) {
                                                coords.add(reader.nextDouble());
                                            }
                                            reader.endArray();
                                            ((LineString) feature.geometry).coordinates.add(coords);
                                        }
                                    }
                                    reader.endArray();
                                }
                            }
                        }
                        reader.endObject();
                    }
                } else {
                    reader.skipValue();
                }
            }
            reader.endObject();
        }
        return feature;
    }

    @Override
    public void write(JsonWriter out, Feature value) throws IOException {
    }
}

class FeatureCollection{
    public String name;
    public String type;
    public List<Feature> features = new ArrayList<>();
}

class Feature<T>{
    public String type;
    public T geometry;
    public Properties properties;
}

class Point {
    public String type = "Point";
    public List<Double> coordinates = new ArrayList<>();
}

class LineString {
    public String type = "LineString";
    public List<List<Double>> coordinates = new ArrayList<>();
}

class Properties {
}