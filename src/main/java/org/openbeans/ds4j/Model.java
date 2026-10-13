package org.openbeans.ds4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The embeddable LLM library, in the role Jetty plays for HTTP: own the
 * lifecycle, resource limits and thread-safety of the model, expose sessions,
 * and let the host application decide everything else (transport, framing,
 * persistence). One Model per process (one Metal device, one weight mmap,
 * one compiled TileLang pipeline cache).
 */
public final class Model implements AutoCloseable {

    public final Metal metal;
    public final Tl tl;
    public final Gguf gguf;
    public final Tokenizer tokenizer;
    public final int nVocab;
    public final int maxCtx;

    private final ReentrantLock allocLock = new ReentrantLock();
    private int activeSessions = 0;
    private int maxSessions = 16;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private Model(Metal metal, Tl tl, Gguf gguf) {
        this.metal = metal;
        this.tl = tl;
        this.gguf = gguf;
        this.tokenizer = new Tokenizer(gguf, 129280);
        this.nVocab = tokenizer.nVocab;
        // the model's own advertised window (ds4flash: 1M); hard-capped at
        // 2^20 to keep every ctx-derived row count inside int range
        this.maxCtx = (int) Math.min(1L << 20,
                gguf.optLong("deepseek4.context_length", 32768));
    }

    /** Assembles the TileLang kernel source and loads the model. */
    public static Model load(String modelPath) throws Exception {
        return load(modelPath, System.getProperty("ds4j.tunedDir"));
    }

    /** tunedDir: optional dir holding <gpukey>.json (autotune override);
     * null/blank resolves it from the classpath instead. */
    public static Model load(String modelPath, String tunedDir) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, tunedDir);
        Gguf gguf = Gguf.open(modelPath);
        return new Model(metal, tl, gguf);
    }

    public void setMaxSessions(int n) { this.maxSessions = n; }
    public int activeSessions() { synchronized (allocLock) { return activeSessions; } }

    /** Creates a conversation session with its own KV caches and scratch buffers. */
    public Session newSession(int ctx) { return newSession(ctx, ""); }

    /** Session with a fixed system prompt (rendered on the first turn). */
    public Session newSession(int ctx, String system) {
        if (closed.get()) throw new IllegalStateException("model closed");
        allocLock.lock();
        try {
            if (ctx <= 0 || ctx > maxCtx) throw new IllegalArgumentException("ctx must be in [1, " + maxCtx + "]");
            if (activeSessions >= maxSessions) throw new IllegalStateException("session limit reached");
            activeSessions++;
            return new Session(this, ctx, system);
        } finally {
            allocLock.unlock();
        }
    }

    void sessionClosed() {
        allocLock.lock();
        try { activeSessions--; } finally { allocLock.unlock(); }
    }

    public boolean isClosed() { return closed.get(); }

    @Override public void close() {
        if (!closed.compareAndSet(false, true)) return;
        // GPU objects live for the process lifetime; freeing them here would
        // break other sessions' encoded pipelines, so close() only stops new
        // sessions and reports. The OS reclaims everything at exit.
    }

    public record ModelInfo(String name, int nVocab, int maxCtx, int nLayer, String device) {}
    public ModelInfo info() {
        return new ModelInfo(Path.of(gguf.path).getFileName().toString(),
                nVocab, maxCtx, 43, metal.deviceName);
    }
}