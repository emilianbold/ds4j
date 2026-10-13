package org.openbeans.ds4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * One conversation bound to one engine (own KV caches and scratch state).
 * A session processes one generation at a time; concurrent calls serialize
 * on an internal lock. All generation runs on the calling thread — the
 * host application owns threading, exactly like an embedded server core.
 *
 * <p>Turn structure follows the C engine: each turn appends
 * {@code <｜User｜> text <｜Assistant｜> [think marker]}, generation ends at
 * EOS without feeding it, and the assistant turn stays in the cache so the
 * next turn only prefills the new user tokens.
 */
public final class Session implements AutoCloseable {
    // Two-buffer decode (C parity: split_after_layers): encode layers 0..3,
    // commit so the GPU starts, encode the rest while it runs — hides the
    // ~0.6ms/token of CPU launch-encoding behind execution. Verified with the
    // process-wide shared residency set: SplitCheck identical, 75/75 hammer
    // rounds clean under C-server co-residency. (The earlier failures blamed
    // on this were actually caused by a SECOND per-Engine attached residency
    // set — fixed by the shared set.) -Dds4j.noSplit falls back to a single
    // command buffer per token.
    static final boolean SPLIT_DECODE = !Boolean.getBoolean("ds4j.noSplit");

    /** Streaming callbacks for a generation. All methods run on the
     *  generating thread; deltas are always valid UTF-8. */
    public interface Listener {
        /** Reasoning-token text (think mode only), before the answer. */
        default void onThinking(String delta) {}
        /** Answer text delta. */
        default void onText(String delta) {}
        /** Raw sampled token id (markers excluded). */
        default void onToken(int id) {}
        default void onComplete(Completion c) {}
        default void onError(Throwable t) {}
    }

    public static final class AbortedException extends RuntimeException {
        public AbortedException() { super("generation aborted"); }
    }
    public static final class ContextOverflowException extends RuntimeException {
        public ContextOverflowException(String m) { super(m); }
    }

    public final Model model;
    public final int ctx;
    /** opt-in: prompts whose NEW tokens are at most this many stream through
     *  the bit-exact decode path instead of batch prefill (correctness mode
     *  while prefill parity is being tightened; 0 = always batch). */
    int decodePrefillMax = 0;

    /** Exactly the tokens that landed in the KV cache during the last
     *  generate() call (prompt + generated + terminal EOS), maintained on
     *  success AND on abort/exception. Callers tracking cache state for
     *  prefix reuse read this instead of guessing. */
    public final List<Integer> lastFedIds = new ArrayList<>();
    private Engine engine;
    private final Tokenizer tok;
    private final List<Tokenizer.ChatMsg> history = new ArrayList<>();
    /** tokens already fed into the cache (next absolute position). */
    private int cursor = 0;
    private boolean open = true;
    private final ReentrantLock genLock = new ReentrantLock();
    private final AtomicBoolean abortFlag = new AtomicBoolean(false);
    private final String system;
    private long totalPromptTokens = 0, totalGenTokens = 0;

    Session(Model model, int ctx) {
        this(model, ctx, "");
    }

    Session(Model model, int ctx, String system) {
        this.model = model;
        this.ctx = ctx;
        this.tok = model.tokenizer;
        this.system = system == null ? "" : system;
        this.engine = new Engine(model.metal, model.tl, model.gguf, ctx);
    }

    // ---------------- conversation API ----------------

    /** One-shot convenience: sends {@code user} with default sampling. */
    public Completion chat(String user) { return chat(user, GenerateParams.defaults(), null); }

    /** Sends a user turn, generates, records the assistant turn in history. */
    public Completion chat(String user, GenerateParams p) { return chat(user, p, null); }

    public Completion chat(String user, GenerateParams p, Listener l) {
        List<Integer> append = new ArrayList<>();
        if (cursor == 0) {
            // first turn renders the full header: BOS, system, then the turn
            append.add(tok.bosId);
            if (system != null && !system.isEmpty()) append.addAll(tok.encode(system));
        }
        append.add(tok.userId);
        append.addAll(tok.encode(user));
        append.add(tok.assistantId);
        append.add(p.think ? tok.thinkStartId : tok.thinkEndId);
        history.add(Tokenizer.ChatMsg.user(user));
        try {
            Completion c = generate(append, p, l);
            history.add(Tokenizer.ChatMsg.assistant(c.text));
            return c;
        } catch (RuntimeException e) {
            history.remove(history.size() - 1);
            throw e;
        }
    }

    /**
     * Raw API: feeds {@code newTokens} (continuation of whatever is already
     * in the cache), then generates until EOS/stop/limit/abort. The caller
     * is responsible for any chat markers inside {@code newTokens}.
     */
    public Completion generate(List<Integer> newTokens, GenerateParams p, Listener l) {
        genLock.lock();
        abortFlag.set(false);
        long t0 = System.nanoTime();
        GenState st = new GenState();
        int beforeCursor = cursor;
        try {
            if (cursor + newTokens.size() + 1 > ctx)
                throw new ContextOverflowException(
                        "prompt of " + newTokens.size() + " tokens exceeds context ("
                                + cursor + "/" + ctx + " used)");
            checkAbort();
            double prefillMs;
            int startPos = cursor;
            st.promptTokens = newTokens.size();

            // ---- prefill the new tokens ----
            if (decodePrefillMax > 0 && newTokens.size() <= decodePrefillMax) {
                for (int i = 0; i < newTokens.size(); i++) {
                    checkAbort();
                    try (Metal.Batch bb = engine.begin()) {
                        engine.decode(bb, newTokens.get(i), startPos + i,
                                i == newTokens.size() - 1);
                    }
                }
            } else {
                engine.prefillFrom(startPos, newTokens, true, this::abortRequested);
            }
            prefillMs = (System.nanoTime() - t0) / 1e6;
            cursor += st.promptTokens;
            st.prefillMs = prefillMs;

            // ---- generation loop ----
            Sampler sampler = new Sampler(p.seed);
            Detok detok = new Detok(tok);
            StringBuilder text = new StringBuilder();
            StringBuilder think = new StringBuilder();
            StringBuilder full = new StringBuilder();
            Dsml.Tracker tracker = new Dsml.Tracker();
            st.text = text;
            st.think = think;
            boolean inThink = p.think;
            int maxGen = p.maxTokens > 0 ? p.maxTokens : Integer.MAX_VALUE;
            int maxStop = 0;
            for (String s : p.stop) maxStop = Math.max(maxStop, s.length());
            StopFilter sf = new StopFilter(maxStop, p.stop);
            String finish = "stop";

            while (true) {
                checkAbort();
                Metal.Buffer lg = engine.logits;
                // C greedy_tool_syntax: inside DSML tool-call structure the
                // payload is decoded greedily (no RNG draw); string bodies and
                // everything else keep the request sampling.
                boolean greedyTool = p.hasTools && !inThink
                        && tracker.inToolCall() && !tracker.usesPayloadSampling();
                int next = p.temperature > 0 && !greedyTool
                        ? sampler.sample(lg, model.nVocab, p.temperature, p.topK, p.topP, p.minP)
                        : Sampler.argmax(lg, model.nVocab);
                if (next == tok.eosId) { closeTurnWithEos(st); break; }
                if (st.genIds.size() >= maxGen) { finish = "length"; closeTurnWithEos(st); break; }
                if (cursor >= ctx) { finish = "context"; break; }

                boolean wasThink = inThink;
                if (next == tok.thinkEndId) inThink = false;
                st.genIds.add(next);
                if (l != null) l.onToken(next);
                if (next != tok.thinkStartId && next != tok.thinkEndId) {
                    String frag = detok.push(next);
                    String safe = sf.push(frag);
                    if (p.hasTools) {
                        full.append(frag);
                        if (inThink) {
                            tracker.reset();
                            tracker.pos = full.length();
                        } else {
                            if (wasThink) tracker.pos = full.length();
                            tracker.update(full.toString());
                        }
                    }
                    if (System.getProperty("ds4j.debugDsml") != null) {
                        System.err.printf("[dsml] tok=%d state=%d greedy=%s frag=%s%n",
                                next, tracker.decode, greedyTool,
                                frag.replace("\n", "\\n"));
                    }
                    if (!safe.isEmpty()) emit(l, inThink, think, text, safe);
                    if (sf.stopHit() != null) { closeTurnWithEos(st); break; }
                }

                if (SPLIT_DECODE) engine.decodeSplit(next, cursor++, true);
                else { try (Metal.Batch bb = engine.begin()) { engine.decode(bb, next, cursor++, true); } }
                st.fedIds.add(next);
            }
            String tail = detok.flush();
            String safeTail = sf.push(tail);
            if (!safeTail.isEmpty()) emit(l, inThink, think, text, safeTail);
            if (sf.stopHit() != null) {
                finish = "stop";
            } else {
                String held = sf.flush();
                if (!held.isEmpty()) emit(l, inThink, think, text, held);
            }

            totalPromptTokens += st.promptTokens;
            totalGenTokens += st.genIds.size();
            double genMs = (System.nanoTime() - t0) / 1e6 - prefillMs;
            Completion c = new Completion(text.toString(), think.toString(), st.genIds, st.fedIds,
                    st.promptTokens, t0, genMs, prefillMs, finish);
            if (l != null) l.onComplete(c);
            return c;
        } catch (RuntimeException e) {
            st.finish = e instanceof AbortedException ? "abort" : "error";
            double totalMs = (System.nanoTime() - t0) / 1e6;
            Completion c = new Completion(st.text == null ? "" : st.text.toString(),
                    st.think == null ? "" : st.think.toString(), st.genIds, st.fedIds,
                    st.promptTokens, t0, Math.max(0, totalMs - st.prefillMs), st.prefillMs, st.finish);
            if (l != null) l.onComplete(c);
            if (l != null) l.onError(e);
            throw e;
        } finally {
            // ledger of what actually landed in the cache, on every exit path
            lastFedIds.clear();
            int fed = cursor - beforeCursor;
            if (fed > 0) {
                if (fed <= newTokens.size()) {
                    lastFedIds.addAll(newTokens.subList(0, fed));
                } else {
                    lastFedIds.addAll(newTokens);
                    int fedGen = fed - newTokens.size();
                    lastFedIds.addAll(st.fedIds.subList(0, Math.min(fedGen, st.fedIds.size())));
                }
            }
            genLock.unlock();
        }
    }

    private static final class GenState {
        int promptTokens;
        double prefillMs;
        final List<Integer> genIds = new ArrayList<>();
        final List<Integer> fedIds = new ArrayList<>();
        StringBuilder text, think;
        String finish = "abort";
    }

    /** C parity: an assistant turn is always closed by feeding EOS into the cache. */
    void closeTurnWithEos(GenState st) {
        if (cursor < ctx) {
            try (Metal.Batch bb = engine.begin()) {
                engine.decode(bb, tok.eosId, cursor++, false);
            }
            st.fedIds.add(tok.eosId);
        }
    }

    private void emit(Listener l, boolean inThink, StringBuilder think,
                      StringBuilder text, String safe) {
        if (inThink) {
            think.append(safe);
            if (l != null) l.onThinking(safe);
        } else {
            text.append(safe);
            if (l != null) l.onText(safe);
        }
    }

    /**
     * Asks the running generation to stop at the next token boundary.
     * The generate call then throws {@link AbortedException} unless it
     * finishes first. Prefill aborts between chunks.
     */
    /** Prompts with at most {@code n} new tokens run through the decode path. */
    public void setDecodePrefill(int n) { this.decodePrefillMax = n; }

    public void abort() { abortFlag.set(true); }

    private void checkAbort() {
        if (abortFlag.get()) throw new AbortedException();
    }
    boolean abortRequested() { return abortFlag.get(); }

    // ---------------- state ----------------

    /** Tokens currently in the context (prompt history + generated). */
    public int contextUsed() { return cursor; }
    public int contextRemaining() { return ctx - cursor; }
    public long totalPromptTokens() { return totalPromptTokens; }
    public long totalGeneratedTokens() { return totalGenTokens; }
    public List<Tokenizer.ChatMsg> history() { return new ArrayList<>(history); }
    public String systemPrompt() { return system; }

    /**
     * Full re-encode of the recorded history, exactly as a fresh session
     * would see it (assistant thinking is not replayed, C parity).
     */
    public List<Integer> reencodeHistory(boolean think) {
        return tok.encodeChatHistory(historyWithSystem(), think);
    }

    private List<Tokenizer.ChatMsg> historyWithSystem() {
        List<Tokenizer.ChatMsg> out = new ArrayList<>();
        if (system != null && !system.isEmpty()) out.add(Tokenizer.ChatMsg.system(system));
        out.addAll(history);
        return out;
    }

    /** Clears the conversation and the KV caches. History is kept. */
    public void reset() {
        genLock.lock();
        try {
            // reuse the engine's Metal buffers (resetState), matching the
            // C's graph reuse; a fresh Engine per request leaked ~0.5 GB
            // of wired buffers and cost 0.1-0.5s of TTFT
            engine.resetState();
            cursor = 0;
            lastFedIds.clear();
        } finally {
            genLock.unlock();
        }
    }

    /** Drops history AND resets caches — back to a fresh session. */
    public void clear() {
        reset();
        genLock.lock();
        try { history.clear(); } finally { genLock.unlock(); }
    }

    @Override public void close() {
        genLock.lock();
        try {
            if (open) {
                open = false;
                model.sessionClosed();
            }
        } finally {
            genLock.unlock();
        }
    }

    // ---------------- stop-sequence streaming filter ----------------

    /**
     * Holds back the tail of every delta so a stop sequence that straddles
     * a token boundary is never emitted. Emits only text that can no
     * longer be part of a stop match.
     */
    static final class StopFilter {
        private final int maxLen;
        private final List<String> stops;
        private final StringBuilder hold = new StringBuilder();
        private String hit = null;

        StopFilter(int maxLen) { this(maxLen, List.of()); }
        StopFilter(int maxLen, List<String> stops) { this.maxLen = maxLen; this.stops = stops; }

        String push(String piece) {
            hold.append(piece);
            if (maxLen == 0) return drainAll();
            if (hit != null) return "";
            for (String s : stops) {
                int idx = hold.indexOf(s);
                if (idx >= 0) {
                    hit = s;
                    String out = hold.substring(0, idx);
                    hold.setLength(0);
                    return out;
                }
            }
            // emit everything except a tail that could still complete a stop
            int keep = Math.min(hold.length(), maxLen - 1);
            int cut = hold.length() - keep;
            if (cut <= 0) return "";
            String out = hold.substring(0, cut);
            hold.delete(0, cut);
            return out;
        }

        String flush() {
            if (hit != null) return "";
            String out = hold.toString();
            hold.setLength(0);
            return out;
        }

        String stopHit() { return hit; }

        private String drainAll() {
            String out = hold.toString();
            hold.setLength(0);
            return out;
        }
    }
}