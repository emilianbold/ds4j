package org.openbeans.ds4j;

import java.util.List;

/**
 * Result of a completed generation.
 */
public final class Completion {
    public final String text;            // assistant message content (no thinking)
    public final String thinking;       // reasoning text ("" when think off)
    public final List<Integer> tokens;  // generated token ids (markers excluded)
    /** ids actually written to the cache this turn: tokens + terminal EOS
     *  (C parity — the session always closes an assistant turn with EOS). */
    public final List<Integer> fedIds;
    public final int promptTokens;
    public final long startNano;       // wall-clock start of generation
    public final double genMs;         // generation wall time (ms)
    public final double prefillMs;
    public final String finishReason;   // "stop" | "length" | "abort" | "context"

    Completion(String text, String thinking, List<Integer> tokens, int promptTokens,
              long startNano, double genMs, double prefillMs, String finishReason) {
        this(text, thinking, tokens, tokens, promptTokens, startNano, genMs, prefillMs, finishReason);
    }

    Completion(String text, String thinking, List<Integer> tokens, List<Integer> fedIds,
              int promptTokens, long startNano, double genMs, double prefillMs, String finishReason) {
        this.fedIds = fedIds;
        this.text = text;
        this.thinking = thinking;
        this.tokens = tokens;
        this.promptTokens = promptTokens;
        this.startNano = startNano;
        this.genMs = genMs;
        this.prefillMs = prefillMs;
        this.finishReason = finishReason;
    }

    public double tokensPerSecond() {
        return genMs > 0 ? tokens.size() / (genMs / 1000.0) : 0;
    }
}