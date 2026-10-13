package org.openbeans.ds4j;

import java.util.List;

/**
 * Generation parameters per request, mirroring the C server's sampling
 * defaults (temperature 1, top-p 1, min-p 0.05) and OpenAI-style extras.
 */
public final class GenerateParams {

    public float temperature = 1.0f;
    public float topP = 1.0f;
    public float minP = 0.05f;
    /** 0: full-vocab sampling (the C default). */
    public int topK = 0;
    /** <=0 means unlimited (bounded by context). */
    public int maxTokens = 0;
    /** true: the model enters through the think-open marker and the
     *  reasoning stream is reported separately. */
    public boolean think = false;
    public long seed = 42;
    /** true when the request carries tools: DSML tool-call structure decodes greedily. */
    public boolean hasTools;
    /** Stop strings (raw text match against the decoded stream). */
    public List<String> stop = List.of();

    public static GenerateParams defaults() { return new GenerateParams(); }

    public GenerateParams temp(float t) { this.temperature = t; return this; }
    public GenerateParams maxTokens(int n) { this.maxTokens = n; return this; }
    public GenerateParams think(boolean b) { this.think = b; return this; }
    public GenerateParams seed(long s) { this.seed = s; return this; }
    public GenerateParams minP(float p) { this.minP = p; return this; }
    public GenerateParams topP(float p) { this.topP = p; return this; }
    public GenerateParams topK(int k) { this.topK = k; return this; }
}