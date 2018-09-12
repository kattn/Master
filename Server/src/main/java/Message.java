public class Message {
    private Command command;
    private String content;

    @Override
    public String toString() {
        return super.toString();
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public Command getCommand() {
        return command;
    }

    public void setCommand(Command command) {
        this.command = command;
    }
}

enum Command {
    REQUEST_FEATURES, REPSONSE_FEATURES
}