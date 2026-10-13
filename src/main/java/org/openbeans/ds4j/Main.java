package org.openbeans.ds4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * CLI on top of the Model/Session API — the embedder's-eye view of the
 * library. Chat markers, prefill, streaming and sampling all go through
 * {@link Session}.
 */
public final class Main {
    public static void main(String[] args) throws Exception {
        String model = System.getProperty("ds4j.model", "ds4flash.gguf");
        int ctx = Integer.parseInt(System.getProperty("ds4j.ctx", "4096"));
        String system = System.getProperty("ds4j.system", "You are a helpful assistant");

        String prompt = null;
        boolean think = Boolean.parseBoolean(System.getProperty("ds4j.think", "false"));
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-p") && i + 1 < args.length) prompt = args[++i];
            else if (args[i].equals("-f") && i + 1 < args.length)
                prompt = Files.readString(Path.of(args[++i]));
            else if (args[i].equals("--think")) think = true;
        }
        if (prompt == null) prompt = "Say hello and tell me what model you are.";

        long t0 = System.nanoTime();
        try (Model m = Model.load(model)) {
            System.out.printf("[ds4j] %s: device %s, compiled in %.1f s%n",
                    Path.of(m.gguf.path).getFileName(), m.metal.deviceName,
                    (System.nanoTime() - t0) / 1e9);
            try (Session s = m.newSession(ctx)) {
                GenerateParams p = GenerateParams.defaults()
                        .temp(Float.parseFloat(System.getProperty("ds4j.temp", "1.0")))
                        .topP(Float.parseFloat(System.getProperty("ds4j.top_p", "1.0")))
                        .minP(Float.parseFloat(System.getProperty("ds4j.min_p", "0.05")))
                        .maxTokens(Integer.parseInt(System.getProperty("ds4j.max", "256")))
                        .think(think);
                long t1 = System.nanoTime();
                Completion c = s.chat(prompt, p, new Session.Listener() {
                    @Override public void onThinking(String d) { System.out.print(d); System.out.flush(); }
                    @Override public void onText(String d) { System.out.print(d); System.out.flush(); }
                    @Override public void onComplete(Completion cc) {
                        System.out.println();
                        System.out.printf("[ds4j] prefill %d tok %.0f ms, generated %d tok in %.1f s (%.1f tok/s), finish=%s%n",
                                cc.promptTokens, cc.prefillMs, cc.tokens.size(),
                                cc.genMs / 1000.0, cc.tokensPerSecond(), cc.finishReason);
                    }
                });
                if (c.finishReason.equals("abort")) System.out.println("[ds4j] aborted");
            }
        }
    }
}