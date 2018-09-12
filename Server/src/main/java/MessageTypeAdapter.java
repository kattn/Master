import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;

import java.io.IOException;

public class MessageTypeAdapter extends TypeAdapter<Message>{
    @Override
    public Message read(JsonReader reader) throws IOException {
        // the first token is the start object
        JsonToken token = reader.peek();
        Message message = new Message();
        if (token.equals(JsonToken.BEGIN_OBJECT)) {
            reader.beginObject();
            while (!reader.peek().equals(JsonToken.END_OBJECT)) {
                if (reader.peek().equals(JsonToken.NAME)) {
                    String nextName = reader.nextName();

                    // Identify what type of command
                    if (nextName.equals("command")) {
                        String type = reader.nextString();

                        if (type.equals("REQUEST_FEATURE"))
                            message.setCommand(Command.REQUEST_FEATURES);
                        else //Should throw a better exception
                            throw new IOException("Met unhandled command type");
                    }

                    // Read in the content
                    if (nextName.equals("content")) {
                        message.setContent(reader.nextString());;
                    }
                }
            }
            reader.endObject();
        }
        return message;
    }

    @Override
    public void write(JsonWriter out, Message value) throws IOException {
        out.beginObject()
            .name("command").value(value.getCommand().name())
            .name("content").jsonValue(value.getContent())
        .endObject();
    }
}
