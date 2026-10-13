package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** Tokenize a chat-completions request exactly as the server renders it. */
public class GenReqIds {
    public static void main(String[] args) throws Exception {
        String reqPath = args[0];
        String outPath = args[1];
        String raw = Files.readString(Path.of(reqPath));
        @SuppressWarnings("unchecked")
        Map<String, Object> req = (Map<String, Object>) Json.parse(raw);
        Server s = new Server(Model.load(System.getProperty("ds4j.model", "ds4flash.gguf")), 0, "127.0.0.1", 8192);
        boolean think = Server.thinkFor(Json.str(req, "model", null), req);
        String toolSchemas = Server.toolSchemasFor(req, raw);
        List<Integer> want = s.renderChat(req, think, toolSchemas);
        StringBuilder sb = new StringBuilder();
        for (int i : want) sb.append(i).append('\n');
        Files.writeString(Path.of(outPath), sb.toString());
        System.out.println("tokens: " + want.size());
    }
}
