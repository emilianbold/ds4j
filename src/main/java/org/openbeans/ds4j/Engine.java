package org.openbeans.ds4j;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * DeepSeek V4 Flash decode engine in Java. All math runs as TileLang Metal
 * kernels through the tiny bridge; Java orchestrates the graph: GGUF weight
 * binding, hyper-connections, MLA-MQA attention with the raw-ring +
 * compressed caches, the compressor frontiers, the sparse indexer, the
 * hash/biased-top-6 router, the IQ2_XXS/Q2_K MoE, and the output head.
 * Mirrors metal_graph_encode_token_raw_swa() in ds4.c.
 */
public final class Engine {
    // ---- shape (DeepSeek V4 Flash; validated against GGUF metadata) ----
    public final int nLayer = 43, nEmb = 4096, nVocab = 129280, nHead = 64,
            nHeadDim = 512, nRot = 64, nHc = 4, sinkhornIters = 20,
            nExpert = 256, nUsed = 6, ffMid = 2048, hashLayers = 3, swa = 128,
            idxHead = 64, idxDim = 128, idxTopK = 512;
    static final float RMS_EPS = 1e-6f, HC_EPS = 1e-6f, EXPERT_SCALE = 1.5f,
            SWIGLU_CLAMP = 10.0f, ROPE_BASE = 10000.0f, ROPE_SCALE = 16.0f,
            COMP_BASE = 160000.0f, BETA_FAST = 32.0f, BETA_SLOW = 1.0f,
            ORIG_CTX = 65536, NEG_INF = -1.0e30f;

    public final Metal metal;
    public final Tl tl;
    public final Gguf gguf;

    // The 80+ GiB model exceeds both Metal's maxBufferLength and Java's 2 GiB
    // FileChannel.map cap, and heavily-overlapping large views turned out to
    // read corrupted through the GPU. Instead every weight tensor gets its own
    // page-aligned no-copy view (largest tensor ~1.06 GiB), cached by file
    // offset; mb() resolves a tensor to (buffer, inner offset).
    // Process-wide model view cache + residency set: every Engine wraps the
    // same model file at the same offsets, so views are shared. The single
    // MTLResidencySet is primed once (first Engine) and attached to the
    // queue once — a SECOND attached 82 GiB set kills kernel dispatch
    // (observed: second engine in a process returns all-zero outputs).
    static final java.util.HashMap<Long, Tl.B> modelViews = new java.util.HashMap<>();

    public final int ctx, rawCap, rawWindow, compCap4;
    public final int[] compRatio = new int[nLayer];
    public final int[] layerCompCap = new int[nLayer];

    // ---- weights ----
    static final class LW { // per-layer offsets into the model buffer
        long hcAttnFn, hcAttnScale, hcAttnBase, attnNorm, attnQa, attnQaNorm,
             attnQb, attnKv, attnKvANorm, attnSinks, attnOutputA, attnOutputB,
             compApe, compKv, compGate, compNorm, idxAttnQb, idxProj,
             idxCompApe, idxCompKv, idxCompGate, idxCompNorm,
             hcFfnFn, hcFfnScale, hcFfnBase, ffnNorm, ffnGateInp, ffnGateExps,
             ffnUpExps, ffnDownExps, ffnGateShexp, ffnUpShexp, ffnDownShexp,
             tid2eid, expProbsB;
        boolean hasTid2eid;
    }
    final LW[] lw = new LW[nLayer];
    public long tokEmbOff, outHcBase, outHcFn, outHcScale, outNormOff, outOff;

    // ---- graph buffers (shared storage: GPU and Java see the same bytes;
    // public: the org.openbeans.ds4j.debug parity harnesses inspect them directly) ----
    public Metal.Buffer curHc, afterFfnHc, flatHc, hcMix, hcSplit,
            attnCur, attnNorm, qr, qrNorm, q, kv, kvRaw,
            compKvCur, compScCur, idxCompKvCur, idxCompScCur,
            attnCompStage, indexerQ, indexerWeights, indexerScores, compSelected,
            heads, attnLow, attnOut, afterAttnHc, ffnCur, ffnNorm,
            sharedGate, sharedUp, sharedMid, sharedOut,
            routerLogits, routerProbs, routerSelected, routerWeights,
            routedGate, routedUp, routedMid, routedOut,
            outputPre, outputWeights, outputEmb, outputNorm, logits,
            embedRow, flashKv, flashPart, q16, iq2Tables;

    public Metal.Buffer[] rawCache = new Metal.Buffer[nLayer];
    public Metal.Buffer[] attnCompCache = new Metal.Buffer[nLayer]; // f16 rows
    public Metal.Buffer[] attnStateKv = new Metal.Buffer[nLayer];
    public Metal.Buffer[] attnStateScore = new Metal.Buffer[nLayer];
    public Metal.Buffer[] idxCompCache = new Metal.Buffer[nLayer];
    Metal.Buffer[] idxStateKv = new Metal.Buffer[nLayer];
    Metal.Buffer[] idxStateScore = new Metal.Buffer[nLayer];
    int[] nComp = new int[nLayer];
    int[] nIndexComp = new int[nLayer];

    static final int IDX_SPARSE_THRESHOLD = 1024;
    // C: metal_graph_resume_prefill_min_tokens() — a resumed checkpoint
    // extension shorter than this streams token-by-token (decode path);
    // larger suffixes (and every fresh prompt) take the batch path.
    public static final int RESUME_BATCH_MIN_TOKENS =
            Integer.getInteger("ds4j.resumeMin", 4);
    int prefillChunkCap;

    // ---- chunked layer-major prefill buffers (allocated on first use) ----
    int pfCap;
    public Metal.Buffer pfTokens; public Metal.Buffer pfCurHc;
    /** debug: snapshot of pfQ right after the q_b projection (pre norm/rope) */
    public Metal.Buffer dbgQraw, dbgCompKv, dbgCompSc, dbgPtState, dbgPtStage; public int dbgPtEmit;
    public Metal.Buffer tmpUnused0, pfNextHc, pfFlatHc, pfHcMix, pfHcSplit, pfAttnCur, pfAttnNorm,
            pfQr, pfQrNorm, pfQ, pfKvRaw, pfKv, pfCompKv, pfCompSc, pfIndexerQ, pfIndexerW,
            pfHeads, pfAttnLow, pfAttnOut, pfAfterAttnHc, pfFfnCur, pfFfnNorm,
            pfSharedGate, pfSharedUp, pfSharedMid, pfSharedOut,
            pfRouterLogits, pfRouterProbs, pfRouterSel, pfRouterW, pfRouterSum,
            pfRoutedGate, pfRoutedUp, pfRoutedMid, pfRoutedMid16, pfRoutedDown, pfRoutedOut,
            pfRowsIdx, pfEmbedRows;
    // grow-on-demand scratch (sizes vary per chunk)
    public static final class Scratch {
        public Metal.Buffer buf;
        Metal.Buffer get(Metal m, long bytes) {
            if (buf == null || buf.bytes < bytes) buf = m.alloc(bytes);
            return buf;
        }
    }
    // per-chunk mask cache: key -> filled mask. Buffers are never refilled:
    // a later layer's mask with different parameters gets a fresh buffer, so
    // kernels already encoded in the same command buffer (not yet executed)
    // always read the content they were encoded against (C parity: the C
    // keeps replaced mask buffers alive until command completion).
    final java.util.HashMap<Long, Metal.Buffer> maskCache = new java.util.HashMap<>();
    public Scratch pfPoolKv = new Scratch(), pfPoolScore1 = new Scratch();
    final Scratch pfSlots = new Scratch();
    final Scratch pfRefreshKv = new Scratch(), pfRefreshSc = new Scratch();
    public final Scratch pfKvStage = new Scratch(), pfF16Round = new Scratch(),
            pfRoundF32 = new Scratch(), pfScoreApe = new Scratch(), pfApeTile = new Scratch(),
            pfIndexerScores = new Scratch(),
            pfTopk = new Scratch(), pfTopkSorted = new Scratch(), pfQ16 = new Scratch(),
            pfX16 = new Scratch(), pfC = new Scratch(), pfX16G = new Scratch(), pfCG = new Scratch(),
            pfMoeMap = new Scratch(), pfPackedRhs = new Scratch(), pfMoeRhs2 = new Scratch();

    public Engine(Metal metal, Tl tl, Gguf gguf, int ctx) {
        this.metal = metal;
        this.tl = tl;
        this.gguf = gguf;
        this.ctx = ctx;


        validateShape();
        for (int il = 0; il < nLayer; il++) {
            compRatio[il] = il < 2 ? 0 : (il % 2 == 0 ? 4 : 128);
            layerCompCap[il] = compRatio[il] == 0 ? 0 : ctx / compRatio[il] + 2;
        }
        rawWindow = Math.min(swa, ctx);
        prefillChunkCap = Integer.parseInt(System.getProperty("ds4j.prefillChunk", "4096")) & ~3;
        prefillChunkCap = Math.max(4, Math.min(prefillChunkCap, ctx)) & ~3;
        long want = Math.min(rawWindow + (long) prefillChunkCap, ctx);
        rawCap = (int) Math.min(8192, Math.max(rawWindow, (want + 255) / 256 * 256));
        compCap4 = ctx / 4 + 2;
        bindWeights();
        allocGraph();
    }

    void validateShape() {
        check("deepseek4.block_count", nLayer);
        check("deepseek4.embedding_length", nEmb);
        check("deepseek4.vocab_size", nVocab);
        check("deepseek4.attention.head_count", nHead);
        check("deepseek4.attention.key_length", nHeadDim);
        check("deepseek4.attention.q_lora_rank", 1024);
        check("deepseek4.expert_count", nExpert);
        check("deepseek4.expert_used_count", nUsed);
        check("deepseek4.expert_feed_forward_length", ffMid);
        if (gguf.t("token_embd.weight").type() != Gguf.T_F16) throw new RuntimeException("token_embd not F16");
    }

    void check(String key, long want) {
        long got = gguf.optLong(key, -1);
        if (got != want) throw new RuntimeException("metadata " + key + " = " + got + ", expected " + want);
    }

    long off(String name) { return gguf.t(name).absOffset(); }
    long off(String fmt, int il) { return gguf.t(String.format(fmt, il)).absOffset(); }

    void bindWeights() {
        tokEmbOff = off("token_embd.weight");
        outHcBase = off("output_hc_base.weight");
        outHcFn = off("output_hc_fn.weight");
        outHcScale = off("output_hc_scale.weight");
        outNormOff = off("output_norm.weight");
        outOff = off("output.weight");
        for (int il = 0; il < nLayer; il++) {
            LW l = new LW();
            l.hcAttnFn = off("blk.%d.hc_attn_fn.weight", il);
            l.hcAttnScale = off("blk.%d.hc_attn_scale.weight", il);
            l.hcAttnBase = off("blk.%d.hc_attn_base.weight", il);
            l.attnNorm = off("blk.%d.attn_norm.weight", il);
            l.attnQa = off("blk.%d.attn_q_a.weight", il);
            l.attnQaNorm = off("blk.%d.attn_q_a_norm.weight", il);
            l.attnQb = off("blk.%d.attn_q_b.weight", il);
            l.attnKv = off("blk.%d.attn_kv.weight", il);
            l.attnKvANorm = off("blk.%d.attn_kv_a_norm.weight", il);
            l.attnSinks = off("blk.%d.attn_sinks.weight", il);
            l.attnOutputA = off("blk.%d.attn_output_a.weight", il);
            l.attnOutputB = off("blk.%d.attn_output_b.weight", il);
            l.hcFfnFn = off("blk.%d.hc_ffn_fn.weight", il);
            l.hcFfnScale = off("blk.%d.hc_ffn_scale.weight", il);
            l.hcFfnBase = off("blk.%d.hc_ffn_base.weight", il);
            l.ffnNorm = off("blk.%d.ffn_norm.weight", il);
            l.ffnGateInp = off("blk.%d.ffn_gate_inp.weight", il);
            l.ffnGateExps = off("blk.%d.ffn_gate_exps.weight", il);
            l.ffnUpExps = off("blk.%d.ffn_up_exps.weight", il);
            l.ffnDownExps = off("blk.%d.ffn_down_exps.weight", il);
            l.ffnGateShexp = off("blk.%d.ffn_gate_shexp.weight", il);
            l.ffnUpShexp = off("blk.%d.ffn_up_shexp.weight", il);
            l.ffnDownShexp = off("blk.%d.ffn_down_shexp.weight", il);
            if (compRatio[il] != 0) {
                l.compApe = off("blk.%d.attn_compressor_ape.weight", il);
                l.compKv = off("blk.%d.attn_compressor_kv.weight", il);
                l.compGate = off("blk.%d.attn_compressor_gate.weight", il);
                l.compNorm = off("blk.%d.attn_compressor_norm.weight", il);
            }
            if (compRatio[il] == 4) {
                l.idxAttnQb = off("blk.%d.indexer.attn_q_b.weight", il);
                l.idxProj = off("blk.%d.indexer.proj.weight", il);
                l.idxCompApe = off("blk.%d.indexer_compressor_ape.weight", il);
                l.idxCompKv = off("blk.%d.indexer_compressor_kv.weight", il);
                l.idxCompGate = off("blk.%d.indexer_compressor_gate.weight", il);
                l.idxCompNorm = off("blk.%d.indexer_compressor_norm.weight", il);
            }
            if (il < hashLayers) {
                l.tid2eid = off("blk.%d.ffn_gate_tid2eid.weight", il);
                l.hasTid2eid = true;
            } else {
                Gguf.Tensor b = gguf.find(String.format("blk.%d.exp_probs_b.bias", il));
                if (b != null && b.type() == Gguf.T_F32) l.expProbsB = b.absOffset();
            }
            lw[il] = l;
        }
    }

    public Metal.Buffer alloc(long bytes) { return metal.alloc(Math.max(1, bytes)); }

    void allocGraph() {
        int sharedDim = (int) gguf.t("blk.0.ffn_gate_shexp.weight").dims()[1];
        curHc = alloc(16384L * 4);
        afterFfnHc = alloc(16384L * 4);
        flatHc = alloc(16384L * 4);
        hcMix = alloc(24L * 4);
        hcSplit = alloc(24L * 4);
        attnCur = alloc(4096L * 4);
        attnNorm = alloc(4096L * 4);
        qr = alloc(1024L * 4);
        qrNorm = alloc(1024L * 4);
        q = alloc(32768L * 4);
        kv = alloc(512L * 4);
        kvRaw = alloc(512L * 4);
        compKvCur = alloc(1024L * 4);
        compScCur = alloc(1024L * 4);
        idxCompKvCur = alloc(256L * 4);
        idxCompScCur = alloc(256L * 4);
        attnCompStage = alloc(4L * 512 * 4); // stage rows; decode uses row 0
        indexerQ = alloc(8192L * 4);
        indexerWeights = alloc(64L * 4);
        indexerScores = alloc((long) compCap4 * 4);
        compSelected = alloc(512L * 4);
        heads = alloc(32768L * 4);
        attnLow = alloc(8192L * 4);
        attnOut = alloc(4096L * 4);
        afterAttnHc = alloc(16384L * 4);
        ffnCur = alloc(4096L * 4);
        ffnNorm = alloc(4096L * 4);
        sharedGate = alloc((long) sharedDim * 4);
        sharedUp = alloc((long) sharedDim * 4);
        sharedMid = alloc((long) sharedDim * 4);
        sharedOut = alloc(4096L * 4);
        routerLogits = alloc(256L * 4);
        routerProbs = alloc(256L * 4);
        routerSelected = alloc(6L * 4);
        routerWeights = alloc(6L * 4);
        routedGate = alloc(6L * ffMid * 4);
        routedUp = alloc(6L * ffMid * 4);
        routedMid = alloc(6L * ffMid * 4);
        routedOut = alloc(4096L * 4);
        outputPre = alloc(4L * 4);
        outputWeights = alloc(4L * 4);
        outputEmb = alloc(4096L * 4);
        outputNorm = alloc(4096L * 4);
        logits = alloc((long) nVocab * 4);
        embedRow = alloc(4096L * 4);
        q16 = alloc(64L * 512 * 2);

        // flash-decode scratch
        long maxKeys = (long) rawWindow + compCap4 + 2;
        flashKv = alloc(maxKeys * 512L * 2);
        long nsplitMax = (maxKeys + 31) / 32 + 2;
        long idxSplits = Math.max(1, Math.min(64, (long) Math.ceil((swa + idxTopK) / 16.0)));
        long partBytes = Math.max(nsplitMax, idxSplits) * 64L * 512 * 4 + 256;
        long statsBytes = Math.max(nsplitMax, idxSplits) * 64L * 8;
        flashPart = alloc(partBytes + statsBytes);

        iq2Tables = alloc(10368L);
        buildIq2Tables();
        residencyPrime();

        for (int il = 0; il < nLayer; il++) {
            rawCache[il] = alloc((long) rawCap * nHeadDim * 4);
            if (compRatio[il] != 0) {
                long coff = compRatio[il] == 4 ? 2 : 1;
                long rows = coff * compRatio[il];
                long width = coff * nHeadDim;
                attnCompCache[il] = alloc((long) layerCompCap[il] * 512 * 2); // F16
                attnStateKv[il] = alloc(rows * width * 4);
                attnStateScore[il] = alloc(rows * width * 4);
                fill(attnStateKv[il], 0f);
                fill(attnStateScore[il], NEG_INF);
            }
            if (compRatio[il] == 4) {
                idxCompCache[il] = alloc((long) layerCompCap[il] * idxDim * 4);
                idxStateKv[il] = alloc(8L * 256 * 4);
                idxStateScore[il] = alloc(8L * 256 * 4);
                fill(idxStateKv[il], 0f);
                fill(idxStateScore[il], NEG_INF);
            }
        }
    }

    static void fill(Metal.Buffer b, float v) {
        long n = b.bytes / 4;
        for (long i = 0; i < n; i++) b.setFloat(i * 4, v);
    }

    /** Reset per-conversation state for reuse: compressor row counters and
     *  the rolling compressor states (the C reuses its graph across
     *  requests; re-allocating ~0.5 GB of Metal buffers per request both
     *  costs ~0.1-0.5s TTFT and leaks the old engine's wired buffers).
     *  KV caches need no clear — rows are position-keyed and only
     *  nComp/nIndexComp rows are ever read. */
    public void resetState() {
        for (int il = 0; il < nLayer; il++) {
            nComp[il] = 0;
            nIndexComp[il] = 0;
            if (compRatio[il] != 0) {
                fill(attnStateKv[il], 0f);
                fill(attnStateScore[il], NEG_INF);
            }
            if (compRatio[il] == 4) {
                fill(idxStateKv[il], 0f);
                fill(idxStateScore[il], NEG_INF);
            }
        }
    }

    /** IQ2 tables buffer: grid-as-floats | ksigns | grid words (see ds4_tl_iq2_tables_buffer). */
    void buildIq2Tables() {
        String src = MetalSources.iq2Tables(); // jar resource / repo / explicit
        long[] grid = new long[256];
        int[] ksigns = new int[128];
        Matcher m = Pattern.compile("ds4_ksigns_iq2xs\\[128\\]\\s*=\\s*\\{([^}]*)\\}").matcher(src);
        if (!m.find()) throw new RuntimeException("ksigns table missing");
        int i = 0;
        for (String tok : m.group(1).split("[,\\s]+"))
            if (!tok.isBlank()) ksigns[i++] = Integer.parseInt(tok.trim());
        m = Pattern.compile("ds4_iq2xxs_grid\\[256\\]\\s*=\\s*\\{([^}]*)\\}").matcher(src);
        if (!m.find()) throw new RuntimeException("iq2xxs grid missing");
        i = 0;
        for (String tok : m.group(1).split("[,\\s]+"))
            if (!tok.isBlank()) grid[i++] = Long.parseLong(tok.trim().replace("0x", ""), 16);
        // floats: grid bytes as floats
        for (int g = 0; g < 256; g++)
            for (int j = 0; j < 8; j++)
                iq2Tables.setFloat((g * 8L + j) * 4, (float) ((grid[g] >>> (8 * j)) & 0xff));
        for (int k = 0; k < 128; k++)
            iq2Tables.setByte(8192L + k, (byte) ksigns[k]);
        for (int g = 0; g < 256; g++)
            for (int j = 0; j < 8; j++)
                iq2Tables.setByte(8320L + g * 8 + j, (byte) ((grid[g] >>> (8 * j)) & 0xff));
    }

    // =====================================================================
    // helpers
    // =====================================================================

    /** Resolve a weight tensor (by absolute file offset) to its no-copy view. */
    public Tl.B mb(long absOff) {
        return modelViews.computeIfAbsent(absOff, a -> wrapView(findTensor(a), a));
    }

    Tl.B wrapView(Gguf.Tensor t, long a) {
        final int page = 16384;
        long poff = a & ~((long) page - 1);
        long len = (t.bytes() + (a - poff) + page - 1) / page * page;
        long addr = Bridge.mmapView(gguf.path, poff, len);
        if (addr == 0) throw new RuntimeException("cannot mmap tensor at " + a);
        long handle = Bridge.bufferWrap(metal.device, addr, len);
        if (handle == 0) throw new RuntimeException("bufferWrap failed for tensor at " + a);
        return new Tl.B(handle, a - poff);
    }

    Gguf.Tensor findTensor(long a) {
        for (Gguf.Tensor c : gguf.tensors.values())
            if (c.absOffset() == a) return c;
        throw new RuntimeException("no tensor at offset " + a);
    }

    // ---- model residency (C parity: MTLResidencySet) ----
    // The C registers its complete model with the driver's residency budget
    // BEFORE inference and never mutates the set afterward. Without a set, a
    // co-resident process that registered its model (the C ds4-server)
    // monopolizes the budget and our unregistered pages become eviction
    // targets — GPU reads then fault mid-kernel and corrupt results (rare,
    // timing-dependent). Empirically, mutating a queue-ATTACHED set (adding
    // allocations after addResidencySet:) breaks multi-command-buffer
    // submission, so the set is populated and committed BEFORE attaching.
    volatile long residencySet;
    boolean metalResidencySupported = true;
    static boolean residencyPrimed = false;

    /** Wrap every model tensor view and register the complete set once
     *  (process-wide; later Engines share the primed views). */
    void residencyPrime() {
        if (Boolean.getBoolean("ds4j.noResidency")) return;
        synchronized (Engine.class) {
            if (residencyPrimed) return;
            residencyPrimed = true;
            try {
                long set = Bridge.residencySetCreate(metal.device, gguf.tensors.size());
                if (set == 0) { metalResidencySupported = false; return; }
                for (Gguf.Tensor t : gguf.tensors.values()) {
                    Tl.B b = modelViews.computeIfAbsent(t.absOffset(),
                            a -> wrapView(t, a));
                    Bridge.residencyAdd(set, b.bufHandle());
                }
                Bridge.residencyCommit(set);
                if (!Boolean.getBoolean("ds4j.noQueueResidency"))
                    Bridge.residencyQueueAdd(metal.queue, set);
                residencySet = set;
            } catch (Throwable t) {
                // driver without residency support: fall back to unregistered
                metalResidencySupported = false;
            }
        }
    }

    /** No-op retained for the generation entries (the set is primed once at
     *  construction and never mutated). */
    void residencyCommitIfDirty() { }

    int ceilDiv(long a, long b) { return (int) ((a + b - 1) / b); }

    /** one command buffer + encoder for the whole token */
    public Metal.Batch begin() { return metal.begin(); }

    // generic gemv grid helper
    int gemvGridX(Tl.Spec s, long outDim) { return (int) (outDim / tl.div(s)); }

    // =====================================================================
    // decode one token; returns logits segment (valid until next decode)
    // =====================================================================
    public int debugMaxLayers = -1;

    // env-gated decode-phase timing (encode vs GPU wait), for perf analysis
    static final boolean DECODE_TIMING = Boolean.getBoolean("ds4j.decodeTiming");
    static final int SPLIT_K = Integer.getInteger("ds4j.splitK", -1);
    public long decodeEncodeNanos, decodeWaitNanos;
    public double decodeGpuMs;

    public Metal.Buffer decode(Metal.Batch bb, int token, int pos, boolean wantLogits) {
        residencyCommitIfDirty();
        long decodeT0 = DECODE_TIMING ? System.nanoTime() : 0L;
        final int rawRow = (int) (pos % rawCap);
        final int nRaw = (int) Math.min(rawWindow, Math.min(pos + 1L, rawCap));
        final int rawStart = (int) (((pos + 1L - nRaw)) % rawCap);

        // ---- embedding: get_rows f16 row -> repeat into 4 HC streams ----
        tl.launch(bb, tl.GET_ROWS_F16_F32, ceilDiv(nEmb, 1024), 1, 1,
                tl.args(tl.GET_ROWS_F16_F32)
                        .i(nEmb / 4).i(1).i(nEmb / 4).i(nEmb),
                Tl.B.b(embedRow, 0), Tl.B.inline(token), mb(tokEmbOff));
        tl.launch(bb, tl.REPEAT_ROWS, ceilDiv(nEmb, 1024), nHc, 1,
                tl.args(tl.REPEAT_ROWS)
                        .i(nEmb / 4).i(nHc * nEmb / 4).i(nHc).i(1).i(nEmb / 4).i(nEmb),
                Tl.B.b(curHc, 0), Tl.B.b(embedRow, 0));

        int layers = debugMaxLayers < 0 ? nLayer : debugMaxLayers;
        for (int il = 0; il < layers; il++) {
            decodeLayer(bb, il, token, pos, rawRow, nRaw, rawStart);
            Metal.Buffer tmp = curHc;
            curHc = afterFfnHc;
            afterFfnHc = tmp;
        }

        if (wantLogits) {
            outputHead(bb);
        }
        if (DECODE_TIMING) decodeEncodeNanos += System.nanoTime() - decodeT0;
        bb.commit();
        long waitT0 = System.nanoTime();
        bb.waitGpu();
        if (DECODE_TIMING) decodeWaitNanos += System.nanoTime() - waitT0;
        return wantLogits ? logits : null;
    }

    /** Decode with the encoding split across two command buffers (C
     *  parity: metal_graph_token_adaptive split_after_layers): buffer 1
     *  carries the first layers and is committed early so the GPU starts
     *  while the CPU encodes the rest — the per-token Java encode time
     *  (~0.7ms) hides behind execution instead of serializing it. The
     *  queue is in-order, so buffer 2 still executes after buffer 1. */
    public Metal.Buffer decodeSplit(int token, int pos, boolean wantLogits) {
        residencyCommitIfDirty();
        long decodeT0 = DECODE_TIMING ? System.nanoTime() : 0L;
        final int rawRow = (int) (pos % rawCap);
        final int nRaw = (int) Math.min(rawWindow, Math.min(pos + 1L, rawCap));
        final int rawStart = (int) (((pos + 1L - nRaw)) % rawCap);
        int layers = debugMaxLayers < 0 ? nLayer : debugMaxLayers;
        int split = SPLIT_K >= 0 ? Math.min(SPLIT_K, Math.max(1, layers - 1))
                : Math.min(4, Math.max(1, layers - 1));

        Metal.Batch bb1 = metal.begin();
        decodeEmbed(bb1, token);
        for (int il = 0; il < split; il++) {
            decodeLayer(bb1, il, token, pos, rawRow, nRaw, rawStart);
            Metal.Buffer tmp = curHc;
            curHc = afterFfnHc;
            afterFfnHc = tmp;
        }
        bb1.commit();

        Metal.Batch bb2 = metal.begin();
        for (int il = split; il < layers; il++) {
            decodeLayer(bb2, il, token, pos, rawRow, nRaw, rawStart);
            Metal.Buffer tmp = curHc;
            curHc = afterFfnHc;
            afterFfnHc = tmp;
        }
        if (wantLogits) {
            outputHead(bb2);
        }
        if (DECODE_TIMING) decodeEncodeNanos += System.nanoTime() - decodeT0;
        bb2.commit();
        long waitT0 = System.nanoTime();
        bb2.waitGpu();
        if (DECODE_TIMING) decodeWaitNanos += System.nanoTime() - waitT0;
        if (DECODE_TIMING) decodeGpuMs += (bb1.gpuTimeMs() + bb2.gpuTimeMs());
        try { bb1.close(); } catch (Exception ignore) {}
        try { bb2.close(); } catch (Exception ignore) {}
        return wantLogits ? logits : null;
    }

    void decodeEmbed(Metal.Batch bb, int token) {
        tl.launch(bb, tl.GET_ROWS_F16_F32, ceilDiv(nEmb, 1024), 1, 1,
                tl.args(tl.GET_ROWS_F16_F32)
                        .i(nEmb / 4).i(1).i(nEmb / 4).i(nEmb),
                Tl.B.b(embedRow, 0), Tl.B.inline(token), mb(tokEmbOff));
        tl.launch(bb, tl.REPEAT_ROWS, ceilDiv(nEmb, 1024), nHc, 1,
                tl.args(tl.REPEAT_ROWS)
                        .i(nEmb / 4).i(nHc * nEmb / 4).i(nHc).i(1).i(nEmb / 4).i(nEmb),
                Tl.B.b(curHc, 0), Tl.B.b(embedRow, 0));
    }

    void decodeLayer(Metal.Batch bb, int il, int token, int pos, int rawRow, int nRaw, int rawStart) {
        final LW l = lw[il];
        final int ratio = compRatio[il];
        final boolean compressed = ratio != 0;
        final float freqBase = compressed ? COMP_BASE : ROPE_BASE;
        final float freqScale = compressed ? 1f / ROPE_SCALE : 1f;
        final float extFactor = compressed ? 1f : 0f;
        final float attnFactor = compressed
                ? (float) (1.0 / (1.0 + 0.1 * Math.log(1.0 / (double) freqScale))) : 1f;
        final int nCtxOrig = (int) (compressed ? ORIG_CTX : 0);

        // ---- HC attention pre-norm ----
        hcPreNorm(bb, l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm,
                curHc, attnCur, attnNorm);

        // ---- QKV + compressor projections ----
        if (ratio == 4) {
            qkvQuadStore(bb, tl.QKV_QUAD_APE16, il, pos, 1024, 512, 1024, 256, 4);
        } else if (ratio == 128) {
            qkvQuadStore(bb, tl.QKV_QUAD_APE16, il, pos, 1024, 512, 512, 0, 128);
        } else {
            // pair gemv: two q8 gemv dispatches on the same input
            q8Gemv(bb, tl.Q8_GEMV_4096, qr, l.attnQa, attnNorm, 1024, 1, 1);
            q8Gemv(bb, tl.Q8_GEMV_4096, kvRaw, l.attnKv, attnNorm, 512, 1, 1);
        }

        // ---- q norm | kv norm + rope + fp8 round-trip + f16 raw store ----
        tl.launch(bb, tl.QKV_RMSNORM_KV_ROPE_FP8_STORE, 1, 2, 1,
                tl.args(tl.QKV_RMSNORM_KV_ROPE_FP8_STORE)
                        .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(RMS_EPS).f(extFactor)
                        .f(freqBase).f(freqScale).i(nCtxOrig).i(pos).i(rawRow),
                Tl.B.b(kv, 0), mb(l.attnKvANorm), Tl.B.b(kvRaw, 0),
                Tl.B.b(qrNorm, 0), mb(l.attnQaNorm), Tl.B.b(qr, 0),
                Tl.B.b(rawCache[il], 0));

        // ---- q up-projection + per-head norm + rope ----
        q8Gemv(bb, tl.Q8_GEMV_1024, q, l.attnQb, qrNorm, nHead * nHeadDim, 1, 1);
        tl.launch(bb, tl.HEAD_RMSNORM_ROPE, nHead, 1, 1,
                tl.args(tl.HEAD_RMSNORM_ROPE)
                        .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(RMS_EPS).f(extFactor)
                        .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(nHead).i(1).i(pos),
                Tl.B.b(q, 0));

        // ---- compressor updates ----
        // C parity: ds4_gpu_compressor_update_tensor returns before the
        // pool when (pos+1)%ratio != 0 — the per-token state store is fused
        // into the QKV quad kernel, and pooling/finalize/commit only run on
        // emit steps. Pooling every token was pure wasted GPU work (~36
        // extra launches/token, ~10% decode-rate loss).
        final boolean emit = compressed && ((pos + 1) % ratio) == 0;
        if (compressed) {
            final int compRow = nComp[il];
            if (emit) {
                if (ratio == 4) {
                    // pool attention state -> stage row 0 (finalize is deferred)
                    tl.launch(bb, tl.POOL_RATIO4_512, 4, 1, 1, tl.args(tl.POOL_RATIO4_512),
                            Tl.B.b(attnCompStage, 0), Tl.B.b(attnStateKv[il], 0),
                            Tl.B.b(attnStateScore[il], 0));
                    // indexer pool -> index cache row (in place finalize later)
                    tl.launch(bb, tl.POOL_RATIO4_128, 1, 1, 1, tl.args(tl.POOL_RATIO4_128),
                            Tl.B.b(idxCompCache[il], (long) nIndexComp[il] * idxDim * 4),
                            Tl.B.b(idxStateKv[il], 0), Tl.B.b(idxStateScore[il], 0));
                    // one dispatch finalizes both rows + shifts both states
                    tl.launch(bb, tl.COMP_ROW_FINALIZE, 22, 1, 1,
                            tl.args(tl.COMP_ROW_FINALIZE)
                                    .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                                    .f(freqBase).f(freqScale).i(nCtxOrig)
                                    .i(pos + 1 - ratio).f(RMS_EPS),
                            Tl.B.b(attnCompCache[il], (long) compRow * 512 * 2),
                            mb(l.compNorm),
                            Tl.B.b(attnCompStage, 0),
                            Tl.B.b(attnStateKv[il], 0), Tl.B.b(attnStateScore[il], 0),
                            mb(l.idxCompNorm),
                            Tl.B.b(idxCompCache[il], (long) nIndexComp[il] * idxDim * 4),
                            Tl.B.b(idxStateKv[il], 0), Tl.B.b(idxStateScore[il], 0));
                    nIndexComp[il]++;
                } else {
                    // softmax pool over the 128-row state -> stage row 0
                    long s0 = 512, s1 = 1, s2 = 128L * 512;
                    tl.launch(bb, tl.SOFTMAX_POOL, ceilDiv(512, 256), 1, 1,
                            tl.args(tl.SOFTMAX_POOL)
                                    .i(512).i(512).i(s0).i(s1).i(s2).i(1).i(128)
                                    .i(s0).i(s1).i(s2),
                            Tl.B.b(attnCompStage, 0), Tl.B.b(attnStateKv[il], 0),
                            Tl.B.b(attnStateScore[il], 0));
                    // norm + rope + fp8 quantize on stage row, then f16 commit
                    tl.launch(bb, tl.RMS_NORM_W512, 1, 1, 1,
                            tl.args(tl.RMS_NORM_W512).f(RMS_EPS).i(1),
                            Tl.B.b(attnCompStage, 0), mb(l.compNorm), Tl.B.b(attnCompStage, 0));
                    tl.launch(bb, tl.ROPE_TAIL_512_H1, 1, 1, 1,
                            tl.args(tl.ROPE_TAIL_512_H1)
                                    .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                                    .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(1)
                                    .i(pos + 1 - ratio).i(1),
                            Tl.B.b(attnCompStage, 0));
                    tl.launch(bb, tl.FP8_QUANTIZE, 1, 1, 1,
                            tl.args(tl.FP8_QUANTIZE).i(1),
                            Tl.B.b(attnCompStage, 0), Tl.B.b(attnCompStage, 0));
                    tl.launch(bb, tl.CPY_F32_F16, ceilDiv(512, 1024), 1, 1,
                            tl.args(tl.CPY_F32_F16).i(512),
                            Tl.B.b(attnCompCache[il], (long) compRow * 512 * 2),
                            Tl.B.b(attnCompStage, 0));
                }
                nComp[il]++;
            }
        }

        // ---- indexer + sparse top-k (ratio 4, long context) ----
        boolean indexed = false;
        int nSelected = 0;
        if (ratio == 4 && nComp[il] > IDX_SPARSE_THRESHOLD && nIndexComp[il] > idxTopK) {
            indexed = true;
            nSelected = Math.min(idxTopK, nIndexComp[il]);
            f16Gemv(bb, tl.F16_GEMV_1024, indexerQ, l.idxAttnQb, qrNorm, idxHead * idxDim, 1, 1);
            tl.launch(bb, tl.ROPE_TAIL_128_H64, 1, 8, 1,
                    tl.args(tl.ROPE_TAIL_128_H64)
                            .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                            .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(1).i(pos).i(1),
                    Tl.B.b(indexerQ, 0));
            tl.launch(bb, tl.INDEXER_HADAMARD, idxHead, 1, 1,
                    tl.args(tl.INDEXER_HADAMARD).i(idxDim).i(idxHead),
                    Tl.B.b(indexerQ, 0));
            f16Gemv(bb, tl.F16_GEMV_4096, indexerWeights, l.idxProj, attnNorm, idxHead, 1, 1);
            float indexScale = (float) (1.0 / Math.sqrt(idxHead * (double) idxDim));
            int nIdxComp = nIndexComp[il];
            tl.launch(bb, tl.INDEXER_SCORES, ceilDiv(nIdxComp, 32), 1, 1,
                    tl.args(tl.INDEXER_SCORES)
                            .i(nIdxComp).i(nIdxComp).i(1).i(4 * nIdxComp).i(4).f(indexScale),
                    Tl.B.b(idxCompCache[il], 0), Tl.B.b(indexerQ, 0),
                    Tl.B.b(indexerScores, 0), Tl.B.b(indexerWeights, 0));
            tl.launch(bb, tl.TOPK_RADIX, 1, 1, 1,
                    tl.args(tl.TOPK_RADIX).i(1).i(idxTopK).i(nIdxComp),
                    Tl.B.b(compSelected, 0), Tl.B.b(indexerScores, 0));
        }

        // ---- attention core ----
        if (indexed) {
            int nSplits = Math.max(1, Math.min(64, ceilDiv((long) swa + idxTopK, 16)));
            long partBytes = (long) nSplits * nHead * 512 * 4;
            long statsBytes = (long) nSplits * nHead * 8;
            tl.launch(bb, tl.CPY_F32_F16, ceilDiv(nHead * (long) nHeadDim, 1024), 1, 1,
                    tl.args(tl.CPY_F32_F16).i(nHead * nHeadDim),
                    Tl.B.b(q16, 0), Tl.B.b(q, 0));
            tl.launch(bb, tl.INDEXED_ATTN, 1, nHead / 32, nSplits,
                    tl.args(tl.INDEXED_ATTN)
                            .i(nComp[il]).i(nHead).i(nRaw).i(nSplits).i(1).i(pos)
                            .i(ratio).i(rawCap).i(rawStart)
                            .f(1f / (float) Math.sqrt(nHeadDim))
                            .i(0).i(idxTopK).i(rawWindow),
                    Tl.B.b(attnCompCache[il], 0), Tl.B.b(heads, 0),
                    Tl.B.b(flashPart, 0), Tl.B.b(q16, 0), Tl.B.b(rawCache[il], 0),
                    mb(l.attnSinks), Tl.B.b(flashPart, partBytes), Tl.B.b(compSelected, 0));
            tl.launch(bb, tl.FLASH_REDUCE, nHead, 1, 1,
                    tl.args(tl.FLASH_REDUCE)
                            .f(0).f(0).f(0).i(0).f(0).f(0).f(0).i(0).i(nHead).i(nSplits).i(0),
                    Tl.B.b(heads, 0), Tl.B.b(flashPart, 0), mb(l.attnSinks),
                    Tl.B.b(flashPart, partBytes));
            // inverse rope tail on heads, separate dispatch (matches the C's
            // indexed decode: reduce without rope, then rope_tail in place)
            tl.launch(bb, tl.ROPE_TAIL_512_H64, 1, 8, 1,
                    tl.args(tl.ROPE_TAIL_512_H64)
                            .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                            .f(freqBase).f(freqScale).i(1).i(nCtxOrig).i(1).i(pos).i(1),
                    Tl.B.b(heads, 0));
        } else {
            int nCompAttn = compressed ? nComp[il] : 0;
            long nKeys = (long) nRaw + nCompAttn;
            long partBytes = ceilDiv(nKeys, 32) * (long) nHead * 512 * 4;
            long statsBytes = ceilDiv(nKeys, 32) * (long) nHead * 8;
            Tl.B compB = nCompAttn > 0 ? Tl.B.b(attnCompCache[il], 0) : Tl.B.b(flashKv, 0);
            tl.launch(bb, tl.FLASH_KV_STAGE, (nKeys + 1) / 2, 1, 1,
                    tl.args(tl.FLASH_KV_STAGE).i(nCompAttn).i(nRaw).i(rawCap).i(rawStart),
                    compB, Tl.B.b(flashKv, 0), Tl.B.b(rawCache[il], 0));
            tl.launch(bb, tl.FLASH_PARTIAL, ceilDiv(nKeys, 32), nHead / 32, 1,
                    tl.args(tl.FLASH_PARTIAL).i(nHead).i(nKeys)
                            .f(1f / (float) Math.sqrt(nHeadDim)),
                    Tl.B.b(flashKv, 0), Tl.B.b(flashPart, 0), Tl.B.b(q, 0),
                    Tl.B.b(flashPart, partBytes));
            tl.launch(bb, tl.FLASH_REDUCE, nHead, 1, 1,
                    tl.args(tl.FLASH_REDUCE)
                            .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).i(1).f(extFactor)
                            .f(freqBase).f(freqScale).i(nCtxOrig).i(nHead)
                            .i(ceilDiv(nKeys, 32)).i(pos),
                    Tl.B.b(heads, 0), Tl.B.b(flashPart, 0), mb(l.attnSinks),
                    Tl.B.b(flashPart, partBytes));
        }

        // ---- attention output: low-rank a + b + hc expand-add ----
        tl.launch(bb, tl.Q8_GEMV_GROUPED, gemvGridX(tl.Q8_GEMV_GROUPED, 1024), 8, 1,
                tl.args(tl.Q8_GEMV_GROUPED).i(8).i(1024),
                Tl.B.b(attnLow, 0), mb(l.attnOutputA), Tl.B.b(heads, 0));
        tl.launch(bb, tl.Q8_GEMV_HC_EXPAND4, gemvGridX(tl.Q8_GEMV_HC_EXPAND4, nEmb), 1, 1,
                tl.args(tl.Q8_GEMV_HC_EXPAND4).i(nEmb),
                Tl.B.b(attnOut, 0), Tl.B.b(afterAttnHc, 0), Tl.B.b(curHc, 0),
                Tl.B.b(hcSplit, 0), mb(l.attnOutputB), Tl.B.b(attnLow, 0));

        // ---- FFN HC pre-norm ----
        hcPreNorm(bb, l.hcFfnFn, l.hcFfnScale, l.hcFfnBase, l.ffnNorm,
                afterAttnHc, ffnCur, ffnNorm);

        // ---- router + shared expert ----
        f16Gemv(bb, tl.F16_GEMV_4096, routerLogits, l.ffnGateInp, ffnNorm, nExpert, 1, 1);
        q8GateUpSwiglu(bb, l.ffnGateShexp, l.ffnUpShexp);
        if (l.hasTid2eid) {
            // hash routing: probs = sqrt(softplus(logits)); table lookup; weights
            tl.launch(bb, tl.UNARY_ROWS, 1, 1, 1,
                    tl.args(tl.UNARY_ROWS).i(nExpert).i(OP_SOFTPLUS_SQRT).f(0).f(0).i(1).i(nExpert).i(nExpert),
                    Tl.B.b(routerProbs, 0), Tl.B.b(routerLogits, 0));
            tl.launch(bb, tl.GET_ROWS_I32, ceilDiv(nUsed, 32), 1, 1,
                    tl.args(tl.GET_ROWS_I32).i(nUsed).i(1).i(nUsed).i(nUsed),
                    Tl.B.b(routerSelected, 0), Tl.B.inline(token), mb(l.tid2eid));
            tl.launch(bb, tl.ROUTER_WEIGHTS, 1, 1, 1,
                    tl.args(tl.ROUTER_WEIGHTS).i(nExpert).i(1).f(EXPERT_SCALE),
                    Tl.B.b(routerProbs, 0), Tl.B.b(routerSelected, 0), Tl.B.b(routerWeights, 0));
        } else {
            boolean hasBias = l.expProbsB != 0;
            tl.launch(bb, tl.ROUTER_FINALIZE, 1, 1, 1,
                    tl.args(tl.ROUTER_FINALIZE).i(hasBias ? 1 : 0),
                    hasBias ? mb(l.expProbsB) : Tl.B.b(routerProbs, 0),
                    Tl.B.b(routerLogits, 0), Tl.B.b(routerProbs, 0),
                    Tl.B.b(routerSelected, 0), Tl.B.b(routerWeights, 0));
        }

        // ---- routed MoE (IQ2_XXS gate/up + Q2_K down) ----
        tl.launch(bb, tl.MOE_GATE_UP, gemvGridX(tl.MOE_GATE_UP, ffMid), nUsed, 1,
                tl.args(tl.MOE_GATE_UP).f(SWIGLU_CLAMP).i(nUsed).i(ffMid),
                Tl.B.b(routedGate, 0), Tl.B.b(iq2Tables, 0), Tl.B.b(routerSelected, 0),
                Tl.B.b(iq2Tables, 8192), Tl.B.b(routedMid, 0), Tl.B.b(routerWeights, 0),
                Tl.B.b(routedUp, 0), mb(l.ffnGateExps), mb(l.ffnUpExps), Tl.B.b(ffnNorm, 0));
        tl.launch(bb, tl.MOE_DOWN, gemvGridX(tl.MOE_DOWN, nEmb), 1, 1,
                tl.args(tl.MOE_DOWN).i(nUsed).i(nEmb),
                Tl.B.b(routerSelected, 0), Tl.B.b(routedMid, 0), Tl.B.b(routedOut, 0),
                mb(l.ffnDownExps));

        // ---- shared down + routed add + HC expand ----
        tl.launch(bb, tl.Q8_GEMV_SHARED_DOWN_HC_EXPAND4, gemvGridX(tl.Q8_GEMV_SHARED_DOWN_HC_EXPAND4, nEmb), 1, 1,
                tl.args(tl.Q8_GEMV_SHARED_DOWN_HC_EXPAND4).i(nEmb),
                Tl.B.b(sharedOut, 0), Tl.B.b(afterFfnHc, 0), Tl.B.b(afterAttnHc, 0),
                Tl.B.b(routedOut, 0), Tl.B.b(hcSplit, 0), mb(l.ffnDownShexp),
                Tl.B.b(sharedMid, 0));
    }

    static final int OP_SCALE = 0, OP_CLAMP = 1, OP_SQRT = 2, OP_SIGMOID = 3,
            OP_SILU = 4, OP_SOFTPLUS = 5, OP_SOFTPLUS_SQRT = 6, OP_FILL = 7;

    void hcPreNorm(Metal.Batch bb, long fnOff, long scaleOff, long baseOff,
                   long normWOff, Metal.Buffer x, Metal.Buffer out, Metal.Buffer normOut) {
        tl.launch(bb, tl.HC_MIX16384, gemvGridX(tl.HC_MIX16384, 24), 1, 1,
                tl.args(tl.HC_MIX16384).f(RMS_EPS).i(24),
                Tl.B.b(hcMix, 0), mb(fnOff), Tl.B.b(x, 0));
        tl.launch(bb, tl.HC_SPLIT_NORM4, 1, 1, 1,
                tl.args(tl.HC_SPLIT_NORM4).f(HC_EPS).f(RMS_EPS).i(1).i(sinkhornIters),
                mb(baseOff), Tl.B.b(out, 0), Tl.B.b(hcMix, 0), Tl.B.b(normOut, 0),
                mb(scaleOff), Tl.B.b(hcSplit, 0), mb(normWOff), Tl.B.b(x, 0));
    }

    void q8Gemv(Metal.Batch bb, Tl.Spec s, Metal.Buffer out, long wOff,
                Metal.Buffer x, long outDim, int gy, int gz) {
        tl.launch(bb, s, gemvGridX(s, outDim), gy, gz,
                tl.args(s).i(outDim), Tl.B.b(out, 0), mb(wOff), Tl.B.b(x, 0));
    }

    void f16Gemv(Metal.Batch bb, Tl.Spec s, Metal.Buffer out, long wOff,
                 Metal.Buffer x, long outDim, int gy, int gz) {
        tl.launch(bb, s, gemvGridX(s, outDim), gy, gz,
                tl.args(s).i(outDim), Tl.B.b(out, 0), mb(wOff), Tl.B.b(x, 0));
    }

    void q8GateUpSwiglu(Metal.Batch bb, long gateOff, long upOff) {
        long sharedDim = sharedMid.bytes / 4;
        tl.launch(bb, tl.Q8_GATE_UP_SWIGLU, gemvGridX(tl.Q8_GATE_UP_SWIGLU, sharedDim), 1, 1,
                tl.args(tl.Q8_GATE_UP_SWIGLU).f(SWIGLU_CLAMP).i(sharedDim),
                Tl.B.b(sharedGate, 0), Tl.B.b(sharedMid, 0), Tl.B.b(sharedUp, 0),
                mb(gateOff), mb(upOff), Tl.B.b(ffnNorm, 0));
    }

    void qkvQuadStore(Metal.Batch bb, Tl.Spec s, int il, int pos,
                      int qDim, int kvDim, int width0, int width1, int ratio) {
        LW l = lw[il];
        long total = (long) qDim + kvDim + width0 + width1;
        tl.launch(bb, s, total / tl.div(s), 1, 1,
                tl.args(s).i(kvDim).i(pos).i(qDim).i(ratio).i(width0).i(width1),
                mb(l.compApe), mb(width1 != 0 ? l.idxCompApe : l.compApe),
                Tl.B.b(compKvCur, 0), Tl.B.b(width1 != 0 ? idxCompKvCur : compKvCur, 0),
                Tl.B.b(compScCur, 0), Tl.B.b(width1 != 0 ? idxCompScCur : compScCur, 0),
                mb(l.compKv), mb(l.compGate),
                mb(width1 != 0 ? l.idxCompKv : l.compKv), mb(width1 != 0 ? l.idxCompGate : l.compGate),
                Tl.B.b(kvRaw, 0), Tl.B.b(qr, 0), mb(l.attnQa), mb(l.attnKv),
                Tl.B.b(attnStateKv[il], 0), Tl.B.b(attnStateScore[il], 0),
                Tl.B.b(width1 != 0 ? idxStateKv[il] : attnStateKv[il], 0),
                Tl.B.b(width1 != 0 ? idxStateScore[il] : attnStateScore[il], 0),
                Tl.B.b(attnNorm, 0));
    }

    // ---- output head: HC collapse + final norm + unembedding ----
    void outputHead(Metal.Batch bb) {
        outputHead(bb, curHc, 0);
    }

    void outputHead(Metal.Batch bb, Metal.Buffer hcBuf, long hcOff) {
        tl.launch(bb, tl.RMS_NORM_PLAIN16384, 1, 1, 1,
                tl.args(tl.RMS_NORM_PLAIN16384).f(RMS_EPS).i(1),
                Tl.B.b(flatHc, 0), Tl.B.b(hcBuf, hcOff));
        f16Gemv(bb, tl.F16_GEMV_16384, outputPre, outHcFn, flatHc, nHc, 1, 1);
        // weights = sigmoid(pre * scale + base) + eps, via bin/unary chains
        tl.launch(bb, tl.BIN_ROWS, 1, 1, 1,
                tl.args(tl.BIN_ROWS).i(nHc).i(1).i(1).i(1).i(nHc).i(OP_MUL).i(1).i(nHc),
                Tl.B.b(outputPre, 0), mb(outHcScale), Tl.B.b(outputPre, 0));
        tl.launch(bb, tl.BIN_ROWS, 1, 1, 1,
                tl.args(tl.BIN_ROWS).i(nHc).i(nHc).i(1).i(nHc).i(nHc).i(OP_ADD).i(1).i(nHc),
                Tl.B.b(outputPre, 0), mb(outHcBase), Tl.B.b(outputPre, 0));
        tl.launch(bb, tl.UNARY_ROWS, 1, 1, 1,
                tl.args(tl.UNARY_ROWS).i(nHc).i(OP_SIGMOID).f(0).f(0).i(1).i(nHc).i(nHc),
                Tl.B.b(outputPre, 0), Tl.B.b(outputPre, 0));
        tl.launch(bb, tl.UNARY_ROWS, 1, 1, 1,
                tl.args(tl.UNARY_ROWS).i(nHc).i(OP_SCALE).f(1.0).f(HC_EPS).i(1).i(nHc).i(nHc),
                Tl.B.b(outputPre, 0), Tl.B.b(outputPre, 0));
        tl.launch(bb, tl.HC_WEIGHTED_SUM4, 16, 1, 1,
                tl.args(tl.HC_WEIGHTED_SUM4).i(1).i(nHc),
                Tl.B.b(outputEmb, 0), Tl.B.b(outputPre, 0), Tl.B.b(hcBuf, hcOff));
        tl.launch(bb, tl.RMS_NORM_W4096, 1, 1, 1,
                tl.args(tl.RMS_NORM_W4096).f(RMS_EPS).i(1),
                Tl.B.b(outputNorm, 0), mb(outNormOff), Tl.B.b(outputEmb, 0));
        q8Gemv(bb, tl.Q8_GEMV_4096, logits, outOff, outputNorm, nVocab, 1, 1);
    }

    static final int OP_ADD = 0, OP_SUB = 1, OP_MUL = 2, OP_DIV = 3;

    // =====================================================================
    // Chunked layer-major prefill (mirrors metal_graph_prefill_chunked)
    // =====================================================================

    void ensurePf(int cap) {
        if (pfCap >= cap) return;
        pfCap = cap;
        long c = cap;
        pfTokens = alloc(c * 4);
        pfCurHc = alloc(c * 16384 * 4); pfNextHc = alloc(c * 16384 * 4); pfFlatHc = alloc(c * 16384 * 4);
        pfHcMix = alloc(c * 24 * 4); pfHcSplit = alloc(c * 24 * 4);
        pfAttnCur = alloc(c * 4096 * 4); pfAttnNorm = alloc(c * 4096 * 4);
        pfQr = alloc(c * 1024 * 4); pfQrNorm = alloc(c * 1024 * 4);
        pfQ = alloc(c * 32768 * 4);
        pfKvRaw = alloc(c * 512 * 4); pfKv = alloc(c * 512 * 4);
        pfCompKv = alloc(c * 1024 * 4); pfCompSc = alloc(c * 1024 * 4);
        pfIndexerQ = alloc(c * 8192 * 4); pfIndexerW = alloc(c * 64 * 4);
        pfHeads = alloc(c * 32768 * 4); pfAttnLow = alloc(c * 8192 * 4); pfAttnOut = alloc(c * 4096 * 4);
        pfAfterAttnHc = alloc(c * 16384 * 4);
        pfFfnCur = alloc(c * 4096 * 4); pfFfnNorm = alloc(c * 4096 * 4);
        pfSharedGate = alloc(c * 2048 * 4); pfSharedUp = alloc(c * 2048 * 4);
        pfSharedMid = alloc(c * 2048 * 4); pfSharedOut = alloc(c * 4096 * 4);
        pfRouterLogits = alloc(c * 256 * 4); pfRouterProbs = alloc(c * 256 * 4);
        pfRouterSel = alloc(c * 6 * 4); pfRouterW = alloc(c * 6 * 4); pfRouterSum = alloc(c * 4);
        pfRoutedGate = alloc(c * 6 * 2048 * 4); pfRoutedUp = alloc(c * 6 * 2048 * 4);
        pfRoutedMid = alloc(c * 6 * 2048 * 4); pfRoutedMid16 = alloc(c * 6 * 2048 * 2);
        pfRoutedDown = alloc(c * 6 * 4096 * 4); pfRoutedOut = alloc(c * 4096 * 4);
        pfRowsIdx = alloc(c * 4);
        pfEmbedRows = alloc(c * 4096 * 4);
        // the attention-compressor stage must hold a whole chunk's pooled rows
        long stageRows = Math.max(4, (long) cap / 4 + 2);
        if (attnCompStage.bytes < stageRows * 512 * 4) attnCompStage = alloc(stageRows * 512 * 4);
    }

    /** Prefill a whole prompt. Returns the logits buffer of the last token. */
    public Metal.Buffer prefill(java.util.List<Integer> tokens, boolean wantLogits) {
        return prefillFrom(0, tokens, wantLogits);
    }

    /**
     * Prefill at an arbitrary continuation position: positions [startPos,
     * startPos+n) are written into the existing caches, with the compressor
     * state carried over from the tokens already in the cache. This is the
     * multi-turn append path (same machinery as chunk 2+ of a long prompt).
     */
    public Metal.Buffer prefillFrom(int startPos, java.util.List<Integer> tokens, boolean wantLogits) {
        return prefillFrom(startPos, tokens, wantLogits, null);
    }

    public Metal.Buffer prefillFrom(int startPos, java.util.List<Integer> tokens,
                                    boolean wantLogits, java.util.function.BooleanSupplier aborted) {
        residencyCommitIfDirty();
        int n = tokens.size();
        int end = startPos + n;
        // C parity (ds4_session_sync_internal): a FRESH prompt (pos0 == 0)
        // always runs the layer-major batch path — streaming decode prefill is
        // hard-disabled in this build. A checkpoint RESUME streams token-by-
        // token only for a suffix shorter than metal_graph_resume_prefill_
        // min_tokens() (4); larger suffixes take the batch path.
        // ds4j.pfStream forces the token-by-token decode path for everything
        // (escape hatch: parity experiments / low-memory mode — the C never
        // does this for fresh prompts here, so outputs may differ in last-bit
        // rounding from the C's batch path).
        boolean streamAll = Boolean.getBoolean("ds4j.pfStream");
        int minBatch = streamAll ? Integer.MAX_VALUE
                : startPos == 0 ? 1 : RESUME_BATCH_MIN_TOKENS;
        if (n < minBatch || prefillChunkCap < minBatch) {
            if (n == 0) return null;
            // stream token-by-token (decode path)
            Metal.Buffer lg = null;
            for (int pos = startPos; pos < end; pos++) {
                if (aborted != null && aborted.getAsBoolean())
                    throw new Session.AbortedException();
                try (Metal.Batch bb = metal.begin()) {
                    lg = decode(bb, tokens.get(pos - startPos), pos, pos == end - 1 && wantLogits);
                }
            }
            return lg;
        }
        ensurePf(Math.min(prefillChunkCap, n));
        int pos0 = startPos;
        while (pos0 < end) {
            if (aborted != null && aborted.getAsBoolean())
                throw new Session.AbortedException();
            // C parity: a resumed range aligns chunk ends to absolute
            // prefill_cap boundaries (metal_graph_prefill_chunked_range)
            int localCap = prefillChunkCap;
            if (startPos != 0) {
                int mod = pos0 % prefillChunkCap;
                if (mod != 0) localCap = Math.min(localCap, prefillChunkCap - mod);
            }
            int chunk = Math.min(localCap, end - pos0);
            // C parity: every chunk runs the full batch path — the C never
            // falls back to the single-token decode path for a ragged tail
            // (the unaligned compressor prefill path handles n % 4 != 0)
            boolean last = pos0 + chunk >= end;
            runChunk(pos0, pos0 - startPos, chunk, tokens);
            if (last && wantLogits) {
                try (Metal.Batch bb = metal.begin()) {
                    outputHead(bb, pfCurHc, (long) (chunk - 1) * 16384 * 4);
                    bb.commit();
                    bb.waitGpu();
                }
            }
            pos0 += chunk;
        }
        return wantLogits ? logits : null;
    }

    void runChunk(int pos0, int t0, int n, java.util.List<Integer> tokens) {
        maskCache.clear();
        for (int i = 0; i < n; i++) pfTokens.setInt((long) i * 4, tokens.get(t0 + i));
        try (Metal.Batch bb = metal.begin()) {
            // ---- embeddings: get_rows f16 -> f32, replicate into 4 HC streams ----
            tl.launch(bb, tl.GET_ROWS_F16_F32, ceilDiv(nEmb, 1024), n, 1,
                    tl.args(tl.GET_ROWS_F16_F32).i(nEmb / 4).i(n).i(nEmb / 4).i(nEmb),
                    Tl.B.b(pfEmbedRows, 0), Tl.B.b(pfTokens, 0), mb(tokEmbOff));
            tl.launch(bb, tl.REPEAT_ROWS, ceilDiv(nEmb, 1024), nHc, n,
                    tl.args(tl.REPEAT_ROWS).i(nEmb / 4).i(nHc * nEmb / 4).i(nHc).i(n).i(nEmb / 4).i(nEmb),
                    Tl.B.b(pfCurHc, 0), Tl.B.b(pfEmbedRows, 0));
            int layers = debugMaxLayers < 0 ? nLayer : debugMaxLayers;
            for (int il = 0; il < layers; il++) {
                prefillLayer(bb, il, pos0, n);
                Metal.Buffer tmp = pfCurHc; pfCurHc = pfNextHc; pfNextHc = tmp;
            }
            bb.commit();
            bb.waitGpu();
        }
    }

    void prefillLayer(Metal.Batch bb, int il, int pos0, int n) {
        final LW l = lw[il];
        final int ratio = compRatio[il];
        final boolean compressed = ratio != 0;
        final float freqBase = compressed ? COMP_BASE : ROPE_BASE;
        final float freqScale = compressed ? 1f / ROPE_SCALE : 1f;
        final float extFactor = compressed ? 1f : 0f;
        final float attnFactor = compressed
                ? (float) (1.0 / (1.0 + 0.1 * Math.log(1.0 / (double) freqScale))) : 1f;
        final int nCtxOrig = (int) (compressed ? ORIG_CTX : 0);
        final int qRank = 1024, qDim = nHead * nHeadDim;

        // ---- HC attention pre-norm (batch) ----
        prefillHcPreNorm(bb, l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm,
                pfCurHc, pfAttnCur, pfAttnNorm, n);

        // ---- q_a / kv projections, fused norms ----
        q8Tier(bb, 4096, pfQr, l.attnQa, pfAttnNorm, 1024, n);
        q8Tier(bb, 4096, pfKvRaw, l.attnKv, pfAttnNorm, 512, n);
        tl.launch(bb, tl.QKV_RMSNORM, n, 2, 1,
                tl.args(tl.QKV_RMSNORM).f(RMS_EPS).i(n),
                Tl.B.b(pfKv, 0), mb(l.attnKvANorm), Tl.B.b(pfKvRaw, 0),
                Tl.B.b(pfQrNorm, 0), mb(l.attnQaNorm), Tl.B.b(pfQr, 0));
        q8Tier(bb, 1024, pfQ, l.attnQb, pfQrNorm, qDim, n);
        if (dbgQraw == null) dbgQraw = alloc(4096L * 32768 * 4);
        // snapshot pfQ (Qraw) into dbgQraw: 3d copy, 1 plane, n rows of 32768
        tl.launch(bb, tl.CPY_F32_3D, ceilDiv(32768, 1024), n, 1,
                tl.args(tl.CPY_F32_3D)
                        .i(32768).i(32768).i(32768).i(32768).i(n)
                        .i(32768).i(32768).i(32768).i(1),
                Tl.B.b(dbgQraw, 0), Tl.B.b(pfQ, 0));
        tl.launch(bb, tl.HEAD_RMSNORM_ROPE, nHead, n, 1,
                tl.args(tl.HEAD_RMSNORM_ROPE)
                        .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(RMS_EPS).f(extFactor)
                        .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(nHead).i(n).i(pos0),
                Tl.B.b(pfQ, 0));
        tl.launch(bb, tl.ROPE_TAIL_512_H1, n, 1, 1,
                tl.args(tl.ROPE_TAIL_512_H1)
                        .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                        .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(n).i(pos0).i(1),
                Tl.B.b(pfKv, 0));
        tl.launch(bb, tl.FP8_QUANTIZE, n, 1, 1,
                tl.args(tl.FP8_QUANTIZE).i(n),
                Tl.B.b(pfKv, 0), Tl.B.b(pfKv, 0));

        // ---- raw ring store (f16-rounded round trip) ----
        Metal.Buffer round16 = pfF16Round.get(metal, (long) n * 512 * 2);
        Metal.Buffer round32 = pfRoundF32.get(metal, (long) n * 512 * 4);
        tl.launch(bb, tl.CPY_F32_F16, ceilDiv((long) n * 512, 1024), 1, 1,
                tl.args(tl.CPY_F32_F16).i(n * 512),
                Tl.B.b(round16, 0), Tl.B.b(pfKv, 0));
        tl.launch(bb, tl.CPY_F16_F32, ceilDiv((long) n * 512, 1024), 1, 1,
                tl.args(tl.CPY_F16_F32).i(n * 512),
                Tl.B.b(round32, 0), Tl.B.b(round16, 0));
        for (int t = 0; t < n; t++) pfRowsIdx.setInt((long) t * 4, (pos0 + t) % rawCap);
        tl.launch(bb, tl.SET_ROWS_F32, ceilDiv(512, 1024), n, 1,
                tl.args(tl.SET_ROWS_F32).i(512 / 4).i(n).i(512 / 4).i(512),
                Tl.B.b(rawCache[il], 0), Tl.B.b(pfRowsIdx, 0), Tl.B.b(round32, 0));

        // ---- attention compressor + indexer (compressed layers) ----
        int nCompChunk = 0;
        if (compressed) {
            int coff = ratio == 4 ? 2 : 1;
            int compWidth = coff * nHeadDim;
            boolean ape16 = gguf.t(String.format("blk.%d.attn_compressor_ape.weight", il)).type() == Gguf.T_F16;
            f16Tier(bb, 4096, pfCompKv, l.compKv, pfAttnNorm, compWidth, n);
            f16Tier(bb, 4096, pfCompSc, l.compGate, pfAttnNorm, compWidth, n);
            nCompChunk = n / ratio;
            int compBefore = nComp[il];
            boolean zeroPrefix = pos0 == 0;
            boolean aligned = pos0 % 4 == 0 && n % 4 == 0;
            boolean perToken = !zeroPrefix && ratio == 4 && !aligned;
            if (dbgCompKv == null) {
                dbgCompKv = alloc((long) prefillChunkCap * compWidth * 4);
                dbgCompSc = alloc((long) prefillChunkCap * compWidth * 4);
            }
            tl.launch(bb, tl.COPY_ROWS_F32, n, 1, 1,
                    tl.args(tl.COPY_ROWS_F32).i(n).i(compWidth),
                    Tl.B.b(dbgCompKv, 0), Tl.B.b(pfCompKv, 0));
            tl.launch(bb, tl.COPY_ROWS_F32, n, 1, 1,
                    tl.args(tl.COPY_ROWS_F32).i(n).i(compWidth),
                    Tl.B.b(dbgCompSc, 0), Tl.B.b(pfCompSc, 0));
            if (perToken) {
                int headN = pos0 % 4 == 0 ? (n / 4) * 4 : 0;
                if (headN > 0) {
                    // Batch the aligned head through the replay and run the
                    // incremental per-token state machine only for the ragged
                    // remainder (< 4 tokens: it can never complete a pooling
                    // window, so it only stores state rows). Mirrors the C
                    // fix: running a whole ragged tail chunk per token costs
                    // ~5 tiny dispatches per layer per token.
                    compressorReplayR4(bb, attnCompStage, 0, attnStateKv[il], attnStateScore[il],
                            l.compApe, ape16, l.compNorm, nHeadDim, pos0, headN, true,
                            freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                    commitCompStage(bb, il, compBefore, nCompChunk);
                    nComp[il] = compBefore + nCompChunk;
                    refreshR4State(bb, attnStateKv[il], attnStateScore[il], l.compKv, l.compGate,
                            l.compApe, ape16, nHeadDim, compWidth, pos0, headN);
                    compressorPerToken(bb, il, pos0, headN, n, l.compApe, ape16, l.compNorm,
                            nHeadDim, attnStateKv[il], attnStateScore[il], true,
                            freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                } else {
                    // C parity: an unaligned-start nonzero-prefix chunk runs
                    // the incremental per-token state machine, not the batch
                    // pool
                    compressorPerToken(bb, il, pos0, 0, n, l.compApe, ape16, l.compNorm,
                            nHeadDim, attnStateKv[il], attnStateScore[il], true,
                            freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                }
            } else if (zeroPrefix || ratio != 4 || !aligned) {
                compressorPrefillBatch(bb, zeroPrefix, attnCompStage, attnStateKv[il], attnStateScore[il],
                        0, l.compApe, ape16, l.compNorm, nHeadDim, ratio, pos0, n, true,
                        freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                if (nCompChunk > 0) commitCompStage(bb, il, compBefore, nCompChunk);
            } else {
                compressorReplayR4(bb, attnCompStage, 0, attnStateKv[il], attnStateScore[il],
                        l.compApe, ape16, l.compNorm, nHeadDim, pos0, n, true,
                        freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                if (nCompChunk > 0) commitCompStage(bb, il, compBefore, nCompChunk);
            }
            if (!perToken) nComp[il] = compBefore + nCompChunk;
            if (ratio == 4 && n >= 4 && !perToken) {
                refreshR4State(bb, attnStateKv[il], attnStateScore[il], l.compKv, l.compGate,
                        l.compApe, ape16, nHeadDim, compWidth, pos0, n);
            }
        }
        int nCompTotal = compressed ? nComp[il] : 0;

        // ---- indexer chain (ratio 4) ----
        if (ratio == 4) {
            int indexWidth = 2 * idxDim;
            boolean ape16 = gguf.t(String.format("blk.%d.indexer_compressor_ape.weight", il)).type() == Gguf.T_F16;
            f16Tier(bb, 4096, pfCompKv, l.idxCompKv, pfAttnNorm, indexWidth, n);
            f16Tier(bb, 4096, pfCompSc, l.idxCompGate, pfAttnNorm, indexWidth, n);
            f16Tier(bb, 1024, pfIndexerQ, l.idxAttnQb, pfQrNorm, idxHead * idxDim, n);
            tl.launch(bb, tl.ROPE_TAIL_128_H64, n, 8, 1,
                    tl.args(tl.ROPE_TAIL_128_H64)
                            .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                            .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(n).i(pos0).i(1),
                    Tl.B.b(pfIndexerQ, 0));
            tl.launch(bb, tl.INDEXER_HADAMARD, n * idxHead, 1, 1,
                    tl.args(tl.INDEXER_HADAMARD).i(idxDim).i(n * idxHead),
                    Tl.B.b(pfIndexerQ, 0));
            f16Tier(bb, 4096, pfIndexerW, l.idxProj, pfAttnNorm, idxHead, n);
            boolean zeroPrefix = pos0 == 0;
            boolean aligned = pos0 % 4 == 0 && n % 4 == 0;
            int idxChunk = nCompChunk;
            int idxBefore = nIndexComp[il];
            boolean idxPerToken = !zeroPrefix && !aligned;
            if (idxPerToken) {
                int idxHeadN = pos0 % 4 == 0 ? (n / 4) * 4 : 0;
                if (idxHeadN > 0) {
                    // Same head/remainder split as the attention compressor
                    // above: replay-batch the aligned head (pool + hadamard
                    // its rows), refresh the state from the head's last four
                    // tokens, then store the ragged remainder per token (it
                    // can never complete a pooling window).
                    compressorReplayR4(bb, idxCompCache[il], (long) idxBefore * idxDim * 4,
                            idxStateKv[il], idxStateScore[il],
                            l.idxCompApe, ape16, l.idxCompNorm, idxDim, pos0, idxHeadN, false,
                            freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                    if (idxChunk > 0) {
                        tl.launch(bb, tl.INDEXER_HADAMARD, idxChunk, 1, 1,
                                tl.args(tl.INDEXER_HADAMARD).i(idxDim).i(idxChunk),
                                Tl.B.b(idxCompCache[il], (long) idxBefore * idxDim * 4));
                    }
                    nIndexComp[il] = idxBefore + idxChunk;
                    refreshR4State(bb, idxStateKv[il], idxStateScore[il], l.idxCompKv, l.idxCompGate,
                            l.idxCompApe, ape16, idxDim, indexWidth, pos0, idxHeadN);
                    compressorPerToken(bb, il, pos0, idxHeadN, n, l.idxCompApe, ape16, l.idxCompNorm,
                            idxDim, idxStateKv[il], idxStateScore[il], false,
                            freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                } else {
                    // C parity: an unaligned-start nonzero-prefix chunk runs
                    // the incremental per-token state machine, not the batch
                    // pool
                    compressorPerToken(bb, il, pos0, 0, n, l.idxCompApe, ape16, l.idxCompNorm,
                            idxDim, idxStateKv[il], idxStateScore[il], false,
                            freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                }
            } else if (zeroPrefix || !aligned) {
                compressorPrefillBatch(bb, zeroPrefix, idxCompCache[il], idxStateKv[il], idxStateScore[il],
                        idxBefore, l.idxCompApe, ape16, l.idxCompNorm, idxDim, 4, pos0, n, false,
                        freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                if (idxChunk > 0) {
                    tl.launch(bb, tl.INDEXER_HADAMARD, idxChunk, 1, 1,
                            tl.args(tl.INDEXER_HADAMARD).i(idxDim).i(idxChunk),
                            Tl.B.b(idxCompCache[il], (long) idxBefore * idxDim * 4));
                }
            } else {
                // C parity: the indexer replay writes into a VIEW at
                // index_before * head_dim — append, never clobber row 0
                compressorReplayR4(bb, idxCompCache[il], (long) idxBefore * idxDim * 4,
                        idxStateKv[il], idxStateScore[il],
                        l.idxCompApe, ape16, l.idxCompNorm, idxDim, pos0, n, false,
                        freqBase, freqScale, extFactor, attnFactor, nCtxOrig);
                if (idxChunk > 0) {
                    tl.launch(bb, tl.INDEXER_HADAMARD, idxChunk, 1, 1,
                            tl.args(tl.INDEXER_HADAMARD).i(idxDim).i(idxChunk),
                            Tl.B.b(idxCompCache[il], (long) idxBefore * idxDim * 4));
                }
            }
            if (!idxPerToken) {
                nIndexComp[il] = idxBefore + idxChunk;
                if (n >= 4) {
                    refreshR4State(bb, idxStateKv[il], idxStateScore[il], l.idxCompKv, l.idxCompGate,
                            l.idxCompApe, ape16, idxDim, indexWidth, pos0, n);
                }
            }
        }

        // ---- attention core ----
        prefillAttention(bb, il, pos0, n, ratio, nCompTotal);

        // ---- inverse rope on heads + attention output ---- (grid.y = 8 head groups)
        tl.launch(bb, tl.ROPE_TAIL_512_H64, n, 8, 1,
                tl.args(tl.ROPE_TAIL_512_H64)
                        .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                        .f(freqBase).f(freqScale).i(1).i(nCtxOrig).i(n).i(pos0).i(1),
                Tl.B.b(pfHeads, 0));
        groupedOutTier(bb, pfAttnLow, l.attnOutputA, pfHeads, 1024, 8, n);
        q8Tier(bb, 8192, pfAttnOut, l.attnOutputB, pfAttnLow, 4096, n);
        hcExpand4(bb, /*hasAdd*/0, pfAttnOut, pfRoutedOut, pfCurHc, pfAfterAttnHc, n);

        // ---- FFN ----
        prefillHcPreNorm(bb, l.hcFfnFn, l.hcFfnScale, l.hcFfnBase, l.ffnNorm,
                pfAfterAttnHc, pfFfnCur, pfFfnNorm, n);
        f16Tier(bb, 4096, pfRouterLogits, l.ffnGateInp, pfFfnNorm, nExpert, n);
        routerSelectBatch(bb, il, n);
        moeBatch(bb, l, n);
        q8Tier(bb, 4096, pfSharedGate, l.ffnGateShexp, pfFfnNorm, ffMid, n);
        q8Tier(bb, 4096, pfSharedUp, l.ffnUpShexp, pfFfnNorm, ffMid, n);
        tl.launch(bb, tl.SWIGLU_FLAT, ceilDiv((long) n * ffMid, 256), 1, 1,
                tl.args(tl.SWIGLU_FLAT).f(1.0).f(SWIGLU_CLAMP).i((long) n * ffMid),
                Tl.B.b(pfSharedGate, 0), Tl.B.b(pfSharedMid, 0), Tl.B.b(pfSharedUp, 0));
        q8Tier(bb, 2048, pfSharedOut, l.ffnDownShexp, pfSharedMid, 4096, n);
        hcExpand4(bb, /*hasAdd*/1, pfRoutedOut, pfSharedOut, pfAfterAttnHc, pfNextHc, n);
    }

    void prefillHcPreNorm(Metal.Batch bb, long fnOff, long scaleOff, long baseOff, long normWOff,
                         Metal.Buffer x, Metal.Buffer out, Metal.Buffer normOut, int n) {
        if (n >= 256) {
            int div = tl.sel(tl.HC_MIX_GEMM) == 3 ? 32 : 64;
            tl.launch(bb, tl.HC_MIX_GEMM, (n + div - 1) / div, 1, 1,
                    tl.args(tl.HC_MIX_GEMM).f(RMS_EPS).i(n),
                    Tl.B.b(pfHcMix, 0), mb(fnOff), Tl.B.b(x, 0));
        } else {
            tl.launch(bb, tl.RMS_NORM_PLAIN16384, n, 1, 1,
                    tl.args(tl.RMS_NORM_PLAIN16384).f(RMS_EPS).i(n),
                    Tl.B.b(pfFlatHc, 0), Tl.B.b(x, 0));
            f16Tier(bb, 16384, pfHcMix, fnOff, pfFlatHc, mixHc, n);
        }
        // C parity: n==1 (DSpark 1-token verifier batch) skips the fused
        // norm kernel — split + separate weighted rms norm; n>1 fuses.
        if (n == 1) {
            tl.launch(bb, tl.HC_SPLIT4, n, 1, 1,
                    tl.args(tl.HC_SPLIT4).f(HC_EPS).i(n).i(sinkhornIters),
                    mb(baseOff), Tl.B.b(out, 0), Tl.B.b(pfHcMix, 0),
                    mb(scaleOff), Tl.B.b(pfHcSplit, 0), Tl.B.b(x, 0));
            tl.launch(bb, tl.RMS_NORM_W4096, n, 1, 1,
                    tl.args(tl.RMS_NORM_W4096).f(RMS_EPS).i(n),
                    Tl.B.b(normOut, 0), mb(normWOff), Tl.B.b(out, 0));
        } else {
            tl.launch(bb, tl.HC_SPLIT_NORM4, n, 1, 1,
                    tl.args(tl.HC_SPLIT_NORM4).f(HC_EPS).f(RMS_EPS).i(n).i(sinkhornIters),
                    mb(baseOff), Tl.B.b(out, 0), Tl.B.b(pfHcMix, 0), Tl.B.b(normOut, 0),
                    mb(scaleOff), Tl.B.b(pfHcSplit, 0), mb(normWOff), Tl.B.b(x, 0));
        }
    }

    static final int mixHc = 24;

    /** Q8_0 x[rows][inDim] @ w[outDim][inDim]^T tier by row count.
     *  C ladder (ds4_gpu_matmul_q8_0_legacy_tensor, all TL families forced):
     *  n==1 gemv; batch GEMV (batch4 kernel for n<=4) for n<=16 or
     *  (n<=31 && out_dim<4096); GEMM tier from n>=17 with out_dim>=4096
     *  (MID), or n>=32 (GEMM_SMALL). */
    void q8Tier(Metal.Batch bb, int inDim, Metal.Buffer out, long wOff, Metal.Buffer x, int outDim, int n) {
        if (n == 1) {
            Tl.Spec s = q8GemvSpec(inDim);
            tl.launch(bb, s, gemvGridX(s, outDim), 1, 1,
                    tl.args(s).i(outDim), Tl.B.b(out, 0), mb(wOff), Tl.B.b(x, 0));
        } else if (n <= 4) {
            Tl.Spec s = q8Batch4Spec(inDim);
            tl.launch(bb, s, outDim, (n + 3) / 4, 1,
                    tl.args(s).i(n).i(outDim), Tl.B.b(out, 0), mb(wOff), Tl.B.b(x, 0));
        } else if (n <= 16 || (n <= 31 && outDim < 4096)) {
            // C parity: q8 batch GEMV band
            Tl.Spec s = q8BatchSpec(inDim);
            tl.launch(bb, s, outDim, ceilDiv(n, 8), 1,
                    tl.args(s).i(n).i(outDim), Tl.B.b(out, 0), mb(wOff), Tl.B.b(x, 0));
        } else {
            // wide out_dim joins the GEMM tier from 17 tokens (C MID tier);
            // 32+ narrow joins the GEMM-small tier
            gemmTier(bb, false, inDim, out, wOff, x, outDim, n);
        }
    }

    Tl.Spec q8GemvSpec(int inDim) {
        return switch (inDim) {
            case 4096 -> tl.Q8_GEMV_4096; case 1024 -> tl.Q8_GEMV_1024;
            case 8192 -> tl.Q8_GEMV_8192; case 2048 -> tl.Q8_GEMV_4096; // 2048 gemv falls back unused
            default -> throw new UnsupportedOperationException("in_dim " + inDim);
        };
    }
    Tl.Spec q8Batch4Spec(int inDim) {
        return switch (inDim) {
            case 4096 -> tl.Q8_GEMV_BATCH4_4096; case 1024 -> tl.Q8_GEMV_BATCH4_1024;
            case 2048 -> tl.Q8_GEMV_BATCH4_2048; case 8192 -> tl.Q8_GEMV_BATCH4_8192;
            case 12288 -> tl.Q8_GEMV_BATCH4_12288; case 256 -> tl.Q8_GEMV_BATCH4_256;
            default -> throw new UnsupportedOperationException("in_dim " + inDim);
        };
    }
    Tl.Spec q8BatchSpec(int inDim) {
        return switch (inDim) {
            case 4096 -> tl.Q8_GEMV_BATCH_4096; case 1024 -> tl.Q8_GEMV_BATCH_1024;
            case 8192 -> tl.Q8_GEMV_BATCH_8192; case 2048 -> tl.Q8_GEMV_BATCH_2048;
            case 12288 -> tl.Q8_GEMV_BATCH_12288; case 256 -> tl.Q8_GEMV_BATCH_256;
            default -> throw new UnsupportedOperationException("in_dim " + inDim);
        };
    }

    /** F16 x[rows][inDim] @ w[outDim][inDim]^T tier by row count.
     *  C ladder (ds4_gpu_matmul_f16_tensor_impl): the f32-x GEMV tiers
     *  (single for n==1, batch4 for n<=4, batch above) cover n<=16 at any
     *  out_dim, and any n<=4096 with out_dim%64!=0; the f16 GEMM tier
     *  (f32->f16 x conversion first) takes n>=17 with out_dim%64==0
     *  (in_dim 4096/1024 only — in_dim 16384 has no GEMM kernel). */
    void f16Tier(Metal.Batch bb, int inDim, Metal.Buffer out, long wOff, Metal.Buffer x, int outDim, int n) {
        if (n == 1) {
            Tl.Spec s = f16GemvSpec(inDim);
            tl.launch(bb, s, gemvGridX(s, outDim), 1, 1,
                    tl.args(s).i(outDim), Tl.B.b(out, 0), mb(wOff), Tl.B.b(x, 0));
        } else if (n <= 4) {
            Tl.Spec s = f16Batch4Spec(inDim);
            tl.launch(bb, s, outDim, (n + 3) / 4, 1,
                    tl.args(s).i(n).i(outDim), Tl.B.b(out, 0), mb(wOff), Tl.B.b(x, 0));
        } else if (n <= 16 || (n <= 4096 && outDim % 64 != 0) || inDim == 16384) {
            // f32-x batch GEMV: the C's band below the GEMM tier
            Tl.Spec s = f16BatchSpec(inDim);
            tl.launch(bb, s, outDim, ceilDiv(n, 8), 1,
                    tl.args(s).i(n).i(outDim), Tl.B.b(out, 0), mb(wOff), Tl.B.b(x, 0));
        } else {
            gemmTier(bb, true, inDim, out, wOff, x, outDim, n);
        }
    }

    Tl.Spec f16GemvSpec(int inDim) {
        return switch (inDim) {
            case 4096 -> tl.F16_GEMV_4096; case 1024 -> tl.F16_GEMV_1024;
            case 16384 -> tl.F16_GEMV_16384; default -> throw new UnsupportedOperationException();
        };
    }
    Tl.Spec f16Batch4Spec(int inDim) {
        return switch (inDim) {
            case 4096 -> tl.F16_GEMV_BATCH4_4096; case 1024 -> tl.F16_GEMV_BATCH4_1024;
            case 16384 -> tl.F16_GEMV_BATCH4_16384; default -> throw new UnsupportedOperationException();
        };
    }
    Tl.Spec f16BatchSpec(int inDim) {
        return switch (inDim) {
            case 4096 -> tl.F16_GEMV_BATCH_4096; case 1024 -> tl.F16_GEMV_BATCH_1024;
            case 16384 -> tl.F16_GEMV_BATCH_16384; default -> throw new UnsupportedOperationException();
        };
    }

    /** 3-dispatch dense GEMM tier: f32->f16 rows, GEMM, copy-out when padded. */
    void gemmTier(Metal.Batch bb, boolean f16w, int inDim, Metal.Buffer out, long wOff, Metal.Buffer x,
                  int outDim, int n) {
        int mPad = (n + 63) / 64 * 64;
        long x16Bytes = (long) mPad * inDim * 2;
        Metal.Buffer x16 = pfX16.get(metal, x16Bytes);
        Tl.Spec cvt = f16w ? f32ToF16Spec(inDim) : f32ToF16Spec(inDim);
        tl.launch(bb, cvt, mPad, 1, 1,
                tl.args(cvt).i(mPad).i(n),
                Tl.B.b(x, 0), Tl.B.b(x16, 0));
        Tl.Spec gemm = f16w ? (inDim == 4096 ? tl.F16_GEMM_4096
                : inDim == 1024 ? tl.F16_GEMM_1024 : null) : q8GemmSpec(inDim);
        if (gemm == null) throw new UnsupportedOperationException("f16 gemm in_dim " + inDim);
        boolean padded = mPad != n;
        Metal.Buffer c = out;
        if (padded) {
            c = pfC.get(metal, (long) mPad * outDim * 4);
        }
        tl.launch(bb, gemm, outDim / 64, mPad / 64, 1,
                tl.args(gemm).i(mPad).i(outDim),
                Tl.B.b(c, 0), mb(wOff), Tl.B.b(x16, 0));
        if (padded) {
            tl.launch(bb, tl.COPY_ROWS_F32, n, 1, 1,
                    tl.args(tl.COPY_ROWS_F32).i(n).i(outDim),
                    Tl.B.b(out, 0), Tl.B.b(c, 0));
        }
    }

    Tl.Spec f32ToF16Spec(int inDim) {
        return switch (inDim) {
            case 4096 -> tl.F32_TO_F16_4096; case 1024 -> tl.F32_TO_F16_1024;
            case 2048 -> tl.F32_TO_F16_2048; case 8192 -> tl.F32_TO_F16_8192;
            default -> throw new UnsupportedOperationException();
        };
    }
    Tl.Spec q8GemmSpec(int inDim) {
        return switch (inDim) {
            case 4096 -> tl.Q8_GEMM_4096; case 1024 -> tl.Q8_GEMM_1024;
            case 8192 -> tl.Q8_GEMM_8192; case 2048 -> tl.Q8_GEMM_2048;
            default -> throw new UnsupportedOperationException("q8 gemm in_dim " + inDim);
        };
    }

    /** attention output low-rank: heads[n][8][4096] @ out_a grouped tier. */
    void groupedOutTier(Metal.Batch bb, Metal.Buffer low, long wOff, Metal.Buffer heads, int rank, int nGroups, int n) {
        if (n == 1) {
            tl.launch(bb, tl.Q8_GEMV_GROUPED, gemvGridX(tl.Q8_GEMV_GROUPED, rank), nGroups, 1,
                    tl.args(tl.Q8_GEMV_GROUPED).i(nGroups).i(rank),
                    Tl.B.b(low, 0), mb(wOff), Tl.B.b(heads, 0));
        } else if (n < 17) {
            tl.launch(bb, tl.Q8_GEMV_GROUPED_BATCH, rank, nGroups, ceilDiv(n, 8),
                    tl.args(tl.Q8_GEMV_GROUPED_BATCH).i(nGroups).i(n).i(rank),
                    Tl.B.b(low, 0), mb(wOff), Tl.B.b(heads, 0));
        } else {
            int mPad = (n + 63) / 64 * 64;
            long cWidth = (long) nGroups * rank;
            Metal.Buffer x16 = pfX16G.get(metal, (long) mPad * nGroups * 4096 * 2);
            Metal.Buffer c = low;
            boolean padded = mPad != n;
            if (padded) c = pfCG.get(metal, (long) mPad * cWidth * 4);
            tl.launch(bb, tl.F32_TO_F16_4096, mPad * nGroups, 1, 1,
                    tl.args(tl.F32_TO_F16_4096).i(mPad * nGroups).i(n * nGroups),
                    Tl.B.b(heads, 0), Tl.B.b(x16, 0));
            tl.launch(bb, tl.Q8_GEMM_GROUPED, rank / 64, mPad / 64, nGroups,
                    tl.args(tl.Q8_GEMM_GROUPED).i(cWidth).i(mPad).i(nGroups).i(rank),
                    Tl.B.b(c, 0), mb(wOff), Tl.B.b(x16, 0));
            if (padded) {
                tl.launch(bb, tl.COPY_ROWS_F32, n, 1, 1,
                        tl.args(tl.COPY_ROWS_F32).i(n).i(cWidth),
                        Tl.B.b(low, 0), Tl.B.b(c, 0));
            }
        }
    }

    /** batch HC expand: out[t,h,:] = (block+add)·post[h] + Σ_j comb[t][j][h]·residual[t,j,:]. */
    void hcExpand4(Metal.Batch bb, int hasAdd, Metal.Buffer block, Metal.Buffer add,
                   Metal.Buffer residual, Metal.Buffer out, int n) {
        // comb/post are views at fixed +32B/+16B into the per-token split
        // vector; the kernel applies the per-token strides via cs2/ps itself.
        tl.launch(bb, tl.HC_EXPAND4, 16, n, 1,
                tl.args(tl.HC_EXPAND4).i(4).i(24).i(hasAdd).i(n).i(24),
                Tl.B.b(add, 0), Tl.B.b(block, 0),
                Tl.B.b(pfHcSplit, 32),
                Tl.B.b(out, 0), Tl.B.b(pfHcSplit, 16), Tl.B.b(residual, 0));
    }


    // =====================================================================
    // prefill helpers: compressor, attention, router, MoE
    // =====================================================================

    /** score+APE into scratch: dst[t*w+e] = sc[t*w+e] + ape[((pos0+t)%ratio)*w+e]. */
    void scoreWithApe(Metal.Batch bb, Tl.B scB, long apeOff, boolean ape16, int width, int ratio,
                      int pos0, int n) {
        Metal.Buffer tile = pfApeTile.get(metal, (long) n * width * 4);
        Tl.Spec cpy = ape16 ? tl.CPY_TILE_F16_F32 : tl.CPY_TILE_F32_F32;
        tl.launch(bb, cpy, ceilDiv((long) n * width, 256), 1, 1,
                tl.args(cpy).i((long) n * width).i(pos0).i(ratio).i(width),
                Tl.B.b(tile, 0), mb(apeOff));
        Metal.Buffer dst = pfScoreApe.get(metal, (long) n * width * 4);
        tl.launch(bb, tl.ADD2, ceilDiv((long) n * width, 256), 1, 1,
                tl.args(tl.ADD2).i((long) n * width),
                scB, Tl.B.b(tile, 0), Tl.B.b(dst, 0));
    }

    /** state rows[.] <- projected kv/sc rows (kv/sc at given byte offsets), APE'd scores. */
    void setRowsProjected(Metal.Batch bb, Metal.Buffer stateKv, Metal.Buffer stateSc,
                          long kvOff, long scOff, long apeOff, boolean ape16,
                          int width, int ratio, int pos, int[] rows) {
        scoreWithApe(bb, new Tl.B(pfCompSc.handle, scOff), apeOff, ape16, width, ratio, pos, rows.length);
        Metal.Buffer score = pfScoreApe.get(metal, (long) rows.length * width * 4);
        tl.launch(bb, tl.SET_ROWS_F32, ceilDiv(width, 1024), rows.length, 1,
                tl.args(tl.SET_ROWS_F32).i(width / 4).i(rows.length).i(width / 4).i(width),
                Tl.B.b(stateKv, 0), rowsIdxBuffer(rows), Tl.B.b(pfCompKv, kvOff));
        tl.launch(bb, tl.SET_ROWS_F32, ceilDiv(width, 1024), rows.length, 1,
                tl.args(tl.SET_ROWS_F32).i(width / 4).i(rows.length).i(width / 4).i(width),
                Tl.B.b(stateSc, 0), rowsIdxBuffer(rows), Tl.B.b(score, 0));
    }

    Metal.Buffer rowsIdxScratch = null;
    Tl.B rowsIdxBuffer(int[] rows) {
        if (rowsIdxScratch == null || rowsIdxScratch.bytes < (long) rows.length * 4)
            rowsIdxScratch = alloc(rows.length * 4);
        for (int i = 0; i < rows.length; i++) rowsIdxScratch.setInt((long) i * 4, rows[i]);
        return Tl.B.b(rowsIdxScratch, 0);
    }

    /** fill state (kv 0, score -inf) over state_rows x width. */
    void fillState(Metal.Batch bb, Metal.Buffer stateKv, Metal.Buffer stateSc, int width, int stateRows) {
        tl.launch(bb, tl.UNARY_ROWS, ceilDiv(width, 256), stateRows, 1,
                tl.args(tl.UNARY_ROWS).i(width).i(OP_FILL).f(0).f(0).i(stateRows).i(width).i(width),
                Tl.B.b(stateKv, 0), Tl.B.b(stateKv, 0));
        tl.launch(bb, tl.UNARY_ROWS, ceilDiv(width, 256), stateRows, 1,
                tl.args(tl.UNARY_ROWS).i(width).i(OP_FILL).f(0).f(NEG_INF).i(stateRows).i(width).i(width),
                Tl.B.b(stateSc, 0), Tl.B.b(stateSc, 0));
    }

    /**
     * Zero-prefix (and general ratio!=4) compressor chunk:
     * fill state, seed prev/tail rows, score+APE over cutoff, pool, norm, rope (pos_step=ratio), fp8.
     */
    void compressorPrefillBatch(Metal.Batch bb, boolean zeroPrefix, Metal.Buffer target,
                                Metal.Buffer stateKv, Metal.Buffer stateSc, long targetOff,
                                long apeOff, boolean ape16, long normOff,
                                int headDim, int ratio, int pos0, int n, boolean quantFp8,
                                float freqBase, float freqScale, float extFactor, float attnFactor, int nCtxOrig) {
        int coff = ratio == 4 ? 2 : 1;
        int width = coff * headDim;
        int stateRows = coff * ratio;
        int nComp = n / ratio;
        int cutoff = nComp * ratio;
        int rem = n - cutoff;

        fillState(bb, stateKv, stateSc, width, stateRows);
        if (ratio == 4 && cutoff >= 4) {
            int prevStart = cutoff - 4;
            setRowsProjected(bb, stateKv, stateSc,
                    (long) prevStart * width * 4, (long) prevStart * width * 4,
                    apeOff, ape16, width, ratio, pos0 + prevStart, new int[]{0, 1, 2, 3});
        }
        if (rem != 0) {
            int[] rows = new int[rem];
            for (int i = 0; i < rem; i++) rows[i] = ratio == 4 ? ratio + i : i;
            setRowsProjected(bb, stateKv, stateSc,
                    (long) cutoff * width * 4, (long) cutoff * width * 4,
                    apeOff, ape16, width, ratio, pos0 + cutoff, rows);
        }

        if (nComp == 0) return;
        // score+APE over the aligned prefix
        scoreWithApe(bb, Tl.B.b(pfCompSc, 0), apeOff, ape16, width, ratio, pos0, cutoff);

        long hd = headDim;
        if (ratio == 4) {
            long packBytes = (long) nComp * 8 * headDim * 4;
            Metal.Buffer poolKv = pfPoolKv.get(metal, packBytes);
            Metal.Buffer poolSc = pfPoolScore1.get(metal, packBytes);
            tl.launch(bb, tl.UNARY_ROWS, ceilDiv(hd, 256), 8 * nComp, 1,
                    tl.args(tl.UNARY_ROWS).i(hd).i(OP_FILL).f(0).f(0).i(8 * nComp).i(hd).i(hd),
                    Tl.B.b(poolKv, 0), Tl.B.b(poolKv, 0));
            tl.launch(bb, tl.UNARY_ROWS, ceilDiv(hd, 256), 8 * nComp, 1,
                    tl.args(tl.UNARY_ROWS).i(hd).i(OP_FILL).f(0).f(NEG_INF).i(8 * nComp).i(hd).i(hd),
                    Tl.B.b(poolSc, 0), Tl.B.b(poolSc, 0));
            // current-half copies: rows 4..7 of every group
            long pRows = headDim, pPlane = 8 * headDim;
            long sRow = width, sPlane = (long) ratio * width;
            cpy3dFull(bb, pfCompKv, headDim * 4, poolKv, 4 * headDim * 4, headDim, ratio, nComp, sRow, sPlane, pRows, pPlane);
            cpy3dFull(bb, pfScoreApe.get(metal, (long) cutoff * width * 4), headDim * 4, poolSc, 4 * headDim * 4,
                    headDim, ratio, nComp, sRow, sPlane, pRows, pPlane);
            if (nComp > 1) {
                cpy3dFull(bb, pfCompKv, 0, poolKv, 8 * headDim * 4, headDim, ratio, nComp - 1, sRow, sPlane, pRows, pPlane);
                cpy3dFull(bb, pfScoreApe.get(metal, (long) cutoff * width * 4), 0, poolSc, 8 * headDim * 4,
                        headDim, ratio, nComp - 1, sRow, sPlane, pRows, pPlane);
            }
            softmaxPool(bb, target, targetOff, poolKv, poolSc, 8, headDim, nComp);
        } else {
            softmaxPool(bb, target, targetOff, pfCompKv, pfScoreApe.get(metal, (long) cutoff * width * 4),
                    ratio, headDim, nComp);
        }
        // the main compressor norms 512-wide rows; the indexer (headDim 128)
        // uses the 128-specialised kernels (C: rms_norm_weight_f32_128 +
        // rope_tail_f32_128_64_h1 in the ds4 trace)
        Tl.Spec normSpec = headDim == 512 ? tl.RMS_NORM_W512 : tl.RMS_NORM_W128;
        Tl.Spec ropeSpec = headDim == 512 ? tl.ROPE_TAIL_512_H1 : tl.ROPE_TAIL_128_H1;
        tl.launch(bb, normSpec, nComp, 1, 1,
                tl.args(normSpec).f(RMS_EPS).i(nComp),
                Tl.B.b(target, targetOff), mb(normOff), Tl.B.b(target, targetOff));
        tl.launch(bb, ropeSpec, nComp, 1, 1,
                tl.args(ropeSpec)
                        .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                        .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(nComp).i(pos0).i(ratio),
                Tl.B.b(target, targetOff));
        if (quantFp8) {
            tl.launch(bb, tl.FP8_QUANTIZE, nComp, 1, 1,
                    tl.args(tl.FP8_QUANTIZE).i(nComp),
                    Tl.B.b(target, targetOff), Tl.B.b(target, targetOff));
        }
    }

    void unusedCpy3dBody(Metal.Batch bb, Metal.Buffer src, long srcOff, Metal.Buffer dst, long dstOff,
               long cols, long rows, long planes, long srcWidth, long srcRatio) {
    }
    void unusedCpy3d() {}

    void softmaxPool(Metal.Batch bb, Metal.Buffer dst, long dstOff, Metal.Buffer kv, Metal.Buffer score,
                     int pooled, int headDim, int nComp) {
        long s0 = headDim, s1 = 1, s2 = (long) pooled * headDim;
        tl.launch(bb, tl.SOFTMAX_POOL, ceilDiv(headDim, 256), nComp, 1,
                tl.args(tl.SOFTMAX_POOL)
                        .i(headDim).i(headDim).i(s0).i(s1).i(s2).i(nComp).i(pooled)
                        .i(s0).i(s1).i(s2),
                Tl.B.b(dst, dstOff), Tl.B.b(kv, 0), Tl.B.b(score, 0));
    }

    /**
     * C parity: an unaligned nonzero-prefix chunk runs the incremental
     * per-token compressor (ds4_gpu_compressor_update_tensor per token):
     * each token's (ape'd) kv/score row enters state slot 4+pos%4; on group
     * completion the 8-row state pools into the comp row (which includes the
     * previous group as the pool's prev half), then norm/rope/(fp8+commit |
     * qat) finalize it and the state shifts cur->prev.
     */
    /** Tokens [t0, n) of the chunk (rows t0..n-1 of pfCompKv/pfCompSc at
     *  positions pos0+t0..pos0+n-1): the incremental per-token state machine.
     *  t0 = 0 processes the whole chunk; a nonzero t0 runs only the ragged
     *  remainder after the aligned head was replay-batched (the remainder is
     *  smaller than one pooling window, so it can only store state rows). */
    void compressorPerToken(Metal.Batch bb, int il, int pos0, int t0, int n,
                            long apeOff, boolean ape16, long normOff, int headDim,
                            Metal.Buffer stateKv, Metal.Buffer stateSc, boolean mainComp,
                            float freqBase, float freqScale, float extFactor, float attnFactor, int nCtxOrig) {
        int width = 2 * headDim;
        // per-row ape'd scores for the whole chunk (row-identical to the C's
        // per-token tile: the tile kernel maps each token by its own position)
        scoreWithApe(bb, Tl.B.b(pfCompSc, 0), apeOff, ape16, width, 4, pos0, n);
        Metal.Buffer score = pfScoreApe.get(metal, (long) n * width * 4);
        // all state slots uploaded once: the command buffer executes after
        // encoding, so a shared mutated index scratch would be read with its
        // final content by every SET_ROWS
        Metal.Buffer slots = pfSlots.get(metal, (long) n * 4);
        for (int t = t0; t < n; t++) slots.setInt((long) t * 4, 4 + ((pos0 + t) & 3));
        for (int t = t0; t < n; t++) {
            int pos = pos0 + t;
            Tl.B slotIdx = Tl.B.b(slots, (long) t * 4);
            tl.launch(bb, tl.SET_ROWS_F32, ceilDiv(width, 1024), 1, 1,
                    tl.args(tl.SET_ROWS_F32).i(width / 4).i(1).i(width / 4).i(width),
                    Tl.B.b(stateKv, 0), slotIdx, Tl.B.b(pfCompKv, (long) t * width * 4));
            tl.launch(bb, tl.SET_ROWS_F32, ceilDiv(width, 1024), 1, 1,
                    tl.args(tl.SET_ROWS_F32).i(width / 4).i(1).i(width / 4).i(width),
                    Tl.B.b(stateSc, 0), slotIdx, Tl.B.b(score, (long) t * width * 4));
            if ((pos & 3) != 3) continue;
            int compPos = pos + 1 - 4;
            if (mainComp) {
                int compRow = nComp[il];
                tl.launch(bb, tl.POOL_RATIO4_512, 4, 1, 1, tl.args(tl.POOL_RATIO4_512),
                        Tl.B.b(attnCompStage, 0), Tl.B.b(stateKv, 0), Tl.B.b(stateSc, 0));
                tl.launch(bb, tl.RMS_NORM_W512, 1, 1, 1,
                        tl.args(tl.RMS_NORM_W512).f(RMS_EPS).i(1),
                        Tl.B.b(attnCompStage, 0), mb(normOff), Tl.B.b(attnCompStage, 0));
                tl.launch(bb, tl.ROPE_TAIL_512_H1, 1, 1, 1,
                        tl.args(tl.ROPE_TAIL_512_H1)
                                .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                                .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(1).i(compPos).i(1),
                        Tl.B.b(attnCompStage, 0));
                tl.launch(bb, tl.FP8_QUANTIZE, 1, 1, 1,
                        tl.args(tl.FP8_QUANTIZE).i(1),
                        Tl.B.b(attnCompStage, 0), Tl.B.b(attnCompStage, 0));
                tl.launch(bb, tl.CPY_F32_F16, ceilDiv(512, 1024), 1, 1,
                        tl.args(tl.CPY_F32_F16).i(512),
                        Tl.B.b(attnCompCache[il], (long) compRow * 512 * 2), Tl.B.b(attnCompStage, 0));
                if (dbgPtStage != null) {
                    tl.launch(bb, tl.COPY_ROWS_F32, 1, 1, 1,
                            tl.args(tl.COPY_ROWS_F32).i(1).i(512),
                            Tl.B.b(dbgPtStage, (long) dbgPtEmit * 512 * 4), Tl.B.b(attnCompStage, 0));
                }
                nComp[il] = compRow + 1;
            } else {
                int idxRow = nIndexComp[il];
                long rowOff = (long) idxRow * headDim * 4;
                tl.launch(bb, tl.POOL_RATIO4_128, 1, 1, 1, tl.args(tl.POOL_RATIO4_128),
                        Tl.B.b(idxCompCache[il], rowOff), Tl.B.b(stateKv, 0), Tl.B.b(stateSc, 0));
                tl.launch(bb, tl.RMS_NORM_W128, 1, 1, 1,
                        tl.args(tl.RMS_NORM_W128).f(RMS_EPS).i(1),
                        Tl.B.b(idxCompCache[il], rowOff), mb(normOff), Tl.B.b(idxCompCache[il], rowOff));
                tl.launch(bb, tl.ROPE_TAIL_128_H1, 1, 1, 1,
                        tl.args(tl.ROPE_TAIL_128_H1)
                                .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                                .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(1).i(compPos).i(1),
                        Tl.B.b(idxCompCache[il], rowOff));
                tl.launch(bb, tl.INDEXER_HADAMARD, 1, 1, 1,
                        tl.args(tl.INDEXER_HADAMARD).i(headDim).i(1),
                        Tl.B.b(idxCompCache[il], rowOff));
                nIndexComp[il] = idxRow + 1;
            }
            tl.launch(bb, tl.COMPRESSOR_SHIFT4, ceilDiv(width, 256), 1, 1,
                    tl.args(tl.COMPRESSOR_SHIFT4).i(width),
                    Tl.B.b(stateKv, 0), Tl.B.b(stateSc, 0));
            if (mainComp && dbgPtState != null) {
                // per-emit state snapshot (post shift, C dump parity)
                tl.launch(bb, tl.COPY_ROWS_F32, 8, 1, 1,
                        tl.args(tl.COPY_ROWS_F32).i(8).i(width),
                        Tl.B.b(dbgPtState, (long) dbgPtEmit * 8 * width * 4), Tl.B.b(stateKv, 0));
                dbgPtEmit++;
            }
        }
    }

    /** Aligned nonzero ratio-4 chunk: replay seeded from the current state.
     *  targetOff: byte offset of the first row in the target (the indexer
     *  cache appends at index_before; the main compressor stages at 0). */
    void compressorReplayR4(Metal.Batch bb, Metal.Buffer target, long targetOff,
                           Metal.Buffer stateKv, Metal.Buffer stateSc,
                           long apeOff, boolean ape16, long normOff, int headDim, int pos0, int n,
                           boolean quantFp8, float freqBase, float freqScale, float extFactor,
                           float attnFactor, int nCtxOrig) {
        int width = 2 * headDim;
        int nComp = n / 4;
        scoreWithApe(bb, Tl.B.b(pfCompSc, 0), apeOff, ape16, width, 4, pos0, n);
        long packBytes = (long) nComp * 8 * headDim * 4;
        Metal.Buffer poolKv = pfPoolKv.get(metal, packBytes);
        Metal.Buffer poolSc = pfPoolScore1.get(metal, packBytes);
        tl.launch(bb, tl.UNARY_ROWS, ceilDiv(headDim, 256), 8 * nComp, 1,
                tl.args(tl.UNARY_ROWS).i(headDim).i(OP_FILL).f(0).f(0).i(8 * nComp).i(headDim).i(headDim),
                Tl.B.b(poolKv, 0), Tl.B.b(poolKv, 0));
        tl.launch(bb, tl.UNARY_ROWS, ceilDiv(headDim, 256), 8 * nComp, 1,
                tl.args(tl.UNARY_ROWS).i(headDim).i(OP_FILL).f(0).f(NEG_INF).i(8 * nComp).i(headDim).i(headDim),
                Tl.B.b(poolSc, 0), Tl.B.b(poolSc, 0));
        // seed group 0 rows 0..3 from the state (previous group, first halves)
        long pRows = headDim, pPlane = 8 * headDim;
        long sRow = width, sPlane = 4L * width;
        cpy3dFull(bb, stateKv, 0, poolKv, 0, headDim, 4, 1, sRow, sRow * 4, pRows, pPlane);
        cpy3dFull(bb, stateSc, 0, poolSc, 0, headDim, 4, 1, sRow, sRow * 4, pRows, pPlane);
        // current halves -> rows 4..7 of every group
        cpy3dFull(bb, pfCompKv, headDim * 4, poolKv, 4 * headDim * 4, headDim, 4, nComp, sRow, sPlane, pRows, pPlane);
        cpy3dFull(bb, pfScoreApe.get(metal, (long) n * width * 4), headDim * 4, poolSc, 4 * headDim * 4,
                headDim, 4, nComp, sRow, sPlane, pRows, pPlane);
        if (nComp > 1) {
            cpy3dFull(bb, pfCompKv, 0, poolKv, 8 * headDim * 4, headDim, 4, nComp - 1, sRow, sPlane, pRows, pPlane);
            cpy3dFull(bb, pfScoreApe.get(metal, (long) n * width * 4), 0, poolSc, 8 * headDim * 4,
                    headDim, 4, nComp - 1, sRow, sPlane, pRows, pPlane);
        }
        softmaxPool(bb, target, targetOff, poolKv, poolSc, 8, headDim, nComp);
        // headDim-selected finalize (C selects the 512/128 kernel variants
        // inside the replay the same way as the batch pool path)
        Tl.Spec normSpec = headDim == 512 ? tl.RMS_NORM_W512 : tl.RMS_NORM_W128;
        Tl.Spec ropeSpec = headDim == 512 ? tl.ROPE_TAIL_512_H1 : tl.ROPE_TAIL_128_H1;
        tl.launch(bb, normSpec, nComp, 1, 1,
                tl.args(normSpec).f(RMS_EPS).i(nComp),
                Tl.B.b(target, targetOff), mb(normOff), Tl.B.b(target, targetOff));
        tl.launch(bb, ropeSpec, nComp, 1, 1,
                tl.args(ropeSpec)
                        .f(attnFactor).f(BETA_FAST).f(BETA_SLOW).f(extFactor)
                        .f(freqBase).f(freqScale).i(0).i(nCtxOrig).i(nComp).i(pos0).i(4),
                Tl.B.b(target, targetOff));
        if (quantFp8) {
            tl.launch(bb, tl.FP8_QUANTIZE, nComp, 1, 1,
                    tl.args(tl.FP8_QUANTIZE).i(nComp),
                    Tl.B.b(target, targetOff), Tl.B.b(target, targetOff));
        }
        fillState(bb, stateKv, stateSc, width, 8);
        setRowsProjected(bb, stateKv, stateSc,
                (long) (n - 4) * width * 4, (long) (n - 4) * width * 4,
                apeOff, ape16, width, 4, pos0 + n - 4, new int[]{0, 1, 2, 3});
    }

    /**
     * dst[p][r][0:cols] = src[p*srcPlane + r*srcRow + ...] with all strides in
     * ELEMENTS, exactly as ds4_tl_encode_cpy_f32_3d packs them
     * (s0=1, d1=cols per row, d2=dstPlane per plane; vec float4 path).
     */
    void cpy3dFull(Metal.Batch bb, Metal.Buffer src, long srcOff, Metal.Buffer dst, long dstOff,
                   long cols, long rows, long planes, long srcRowStride, long srcPlaneStride,
                   long dstRowStride, long dstPlaneStride) {
        tl.launch(bb, tl.CPY_F32_3D, ceilDiv(cols, 1024), rows, planes,
                tl.args(tl.CPY_F32_3D).i(cols).i(dstRowStride).i(dstPlaneStride)
                        .i(planes).i(rows).i(1).i(srcRowStride).i(srcPlaneStride)
                        .i(1), // vec: float4 fast path (all strides %4 == 0, offsets 16B-aligned)
                Tl.B.b(dst, dstOff), Tl.B.b(src, srcOff));
    }

    /** commit f32 stage rows -> f16 attention comp cache rows [first, first+rows). */
    void commitCompStage(Metal.Batch bb, int il, int first, int rows) {
        tl.launch(bb, tl.CPY_F32_F16, ceilDiv((long) rows * 512, 1024), 1, 1,
                tl.args(tl.CPY_F32_F16).i(rows * 512),
                Tl.B.b(attnCompCache[il], (long) first * 512 * 2), Tl.B.b(attnCompStage, 0));
    }

    /** re-project the chunk's last 4 tokens through the compressor into the state rows 0..3. */
    void refreshR4State(Metal.Batch bb, Metal.Buffer stateKv, Metal.Buffer stateSc,
                        long kvW, long gateW, long apeOff, boolean ape16,
                        int headDim, int width, int pos0, int n) {
        long tailOff = (long) (n - 4) * nEmb * 4;
        // batch4 f16 gemv (4 tokens) from the tail of batch attn_norm; the
        // output stages in a dedicated 4-row scratch (the C keeps the batch
        // comp buffers untouched — the pool already consumed them)
        Metal.Buffer rKv = pfRefreshKv.get(metal, (long) width * 4 * 4);
        Metal.Buffer rSc = pfRefreshSc.get(metal, (long) width * 4 * 4);
        tl.launch(bb, tl.F16_GEMV_BATCH4_4096, width, 1, 1,
                tl.args(tl.F16_GEMV_BATCH4_4096).i(4).i(width),
                Tl.B.b(rKv, 0), mb(kvW), Tl.B.b(pfAttnNorm, tailOff));
        tl.launch(bb, tl.F16_GEMV_BATCH4_4096, width, 1, 1,
                tl.args(tl.F16_GEMV_BATCH4_4096).i(4).i(width),
                Tl.B.b(rSc, 0), mb(gateW), Tl.B.b(pfAttnNorm, tailOff));
        fillState(bb, stateKv, stateSc, width, 8);
        setRowsProjectedR(bb, stateKv, stateSc, rKv, rSc, apeOff, ape16, width, 4, pos0 + n - 4,
                new int[]{0, 1, 2, 3});
    }

    /** state rows[.] <- projected kv/sc rows from a staging scratch. */
    void setRowsProjectedR(Metal.Batch bb, Metal.Buffer stateKv, Metal.Buffer stateSc,
                           Metal.Buffer kvSrc, Metal.Buffer scSrc, long apeOff, boolean ape16,
                           int width, int ratio, int pos, int[] rows) {
        scoreWithApeR(bb, scSrc, apeOff, ape16, width, ratio, pos, rows.length);
        Metal.Buffer score = pfScoreApe.get(metal, (long) rows.length * width * 4);
        tl.launch(bb, tl.SET_ROWS_F32, ceilDiv(width, 1024), rows.length, 1,
                tl.args(tl.SET_ROWS_F32).i(width / 4).i(rows.length).i(width / 4).i(width),
                Tl.B.b(stateKv, 0), rowsIdxBuffer(rows), Tl.B.b(kvSrc, 0));
        tl.launch(bb, tl.SET_ROWS_F32, ceilDiv(width, 1024), rows.length, 1,
                tl.args(tl.SET_ROWS_F32).i(width / 4).i(rows.length).i(width / 4).i(width),
                Tl.B.b(stateSc, 0), rowsIdxBuffer(rows), Tl.B.b(score, 0));
    }

    void scoreWithApeR(Metal.Batch bb, Metal.Buffer sc, long apeOff, boolean ape16, int width, int ratio,
                       int pos0, int n) {
        Metal.Buffer tile = pfApeTile.get(metal, (long) n * width * 4);
        Tl.Spec cpy = ape16 ? tl.CPY_TILE_F16_F32 : tl.CPY_TILE_F32_F32;
        tl.launch(bb, cpy, ceilDiv((long) n * width, 256), 1, 1,
                tl.args(cpy).i((long) n * width).i(pos0).i(ratio).i(width),
                Tl.B.b(tile, 0), mb(apeOff));
        Metal.Buffer dst = pfScoreApe.get(metal, (long) n * width * 4);
        tl.launch(bb, tl.ADD2, ceilDiv((long) n * width, 256), 1, 1,
                tl.args(tl.ADD2).i((long) n * width),
                Tl.B.b(sc, 0), Tl.B.b(tile, 0), Tl.B.b(dst, 0));
    }

    // ---- prefill attention core ----

    void prefillAttention(Metal.Batch bb, int il, int pos0, int n, int ratio, int nComp) {
        boolean zeroPrefix = pos0 == 0;
        float scale = (float) (1.0 / Math.sqrt(nHeadDim));
        int nKeys;
        if (zeroPrefix) {
            if (ratio == 0 || nComp == 0) {
                nKeys = n;
                stageF16(bb, pfKv, 0, pfKvStage, 0, (long) n * nHeadDim);
                Metal.Buffer mask = cachedMask(true, 0, n, nKeys, rawWindow, 0, ratio);
                flashPrefill(bb, n, nKeys, scale, l_sinks(il), mask);
                return;
            }
            if (ratio == 4 && nComp > idxTopK) {
                // indexed: scores -> topk -> sort -> q16 -> indexed attention
                indexerScores(bb, il, pos0, n, nComp, scale);
                topkAndSort(bb, n, nComp);
                Metal.Buffer q16 = pfQ16.get(metal, (long) n * nHead * nHeadDim * 2);
                tl.launch(bb, tl.CPY_F32_F16, ceilDiv((long) n * nHead * nHeadDim, 1024), 1, 1,
                        tl.args(tl.CPY_F32_F16).i(n * nHead * nHeadDim),
                        Tl.B.b(q16, 0), Tl.B.b(pfQ, 0));
                tl.launch(bb, tl.INDEXED_ATTN, n, nHead / 32, 1,
                        tl.args(tl.INDEXED_ATTN)
                                .i(nComp).i(nHead).i(n).i(1).i(n).i(pos0)
                                .i(4).i(rawCap).i(0)
                                .f(scale)
                                .i(1).i(idxTopK).i(rawWindow),
                        Tl.B.b(attnCompCache[il], 0), Tl.B.b(pfHeads, 0), Tl.B.b(pfHeads, 0),
                        Tl.B.b(q16, 0), Tl.B.b(rawCache[il], 0), mb(l_sinks(il)),
                        Tl.B.b(pfHeads, 0), Tl.B.b(pfTopkSorted.get(metal, (long) n * idxTopK * 4), 0));
                return;
            }
            // static mixed: raw f16 rows + comp f16 rows, causal+window+comp-visibility mask
            nKeys = n + nComp;
            Metal.Buffer kvStage = pfKvStage.get(metal, (long) nKeys * nHeadDim * 2);
            tl.launch(bb, tl.CPY_F32_F16, ceilDiv((long) n * nHeadDim, 1024), 1, 1,
                    tl.args(tl.CPY_F32_F16).i(n * nHeadDim),
                    Tl.B.b(kvStage, 0), Tl.B.b(pfKv, 0));
            tl.launch(bb, tl.CPY_U16, ceilDiv((long) nComp * nHeadDim, 1024), 1, 1,
                    tl.args(tl.CPY_U16).i(nComp * nHeadDim),
                    Tl.B.b(kvStage, (long) n * nHeadDim * 2), Tl.B.b(attnCompCache[il], 0));
            Metal.Buffer mask = cachedMask(false, 0, n, nKeys, rawWindow, ratio, nComp);
            flashPrefill(bb, n, nKeys, scale, l_sinks(il), mask);
            return;
        }
        // nonzero prefix: linearized raw window out of the ring (+ comp rows)
        int nRaw = (int) Math.min((long) n + rawWindow, Math.min((long) pos0 + n, rawCap));
        int rawStart = (int) (((long) pos0 + n - nRaw) % rawCap);
        if (ratio == 4 && nComp > idxTopK) {
            indexerScores(bb, il, pos0, n, nComp, scale);
            topkAndSort(bb, n, nComp);
            Metal.Buffer q16 = pfQ16.get(metal, (long) n * nHead * nHeadDim * 2);
            tl.launch(bb, tl.CPY_F32_F16, ceilDiv((long) n * nHead * nHeadDim, 1024), 1, 1,
                    tl.args(tl.CPY_F32_F16).i(n * nHead * nHeadDim),
                    Tl.B.b(q16, 0), Tl.B.b(pfQ, 0));
            tl.launch(bb, tl.INDEXED_ATTN, n, nHead / 32, 1,
                    tl.args(tl.INDEXED_ATTN)
                            .i(nComp).i(nHead).i(nRaw).i(1).i(n).i(pos0)
                            .i(4).i(rawCap).i(rawStart)
                            .f(scale)
                            .i(1).i(idxTopK).i(rawWindow),
                    Tl.B.b(attnCompCache[il], 0), Tl.B.b(pfHeads, 0), Tl.B.b(pfHeads, 0),
                    Tl.B.b(q16, 0), Tl.B.b(rawCache[il], 0), mb(l_sinks(il)),
                    Tl.B.b(pfHeads, 0), Tl.B.b(pfTopkSorted.get(metal, (long) n * idxTopK * 4), 0));
            return;
        }
        boolean mixed = nComp > 0;
        nKeys = nRaw + (mixed ? nComp : 0);
        Metal.Buffer kvStage = pfKvStage.get(metal, (long) nKeys * nHeadDim * 2);
        stageRingF16(bb, rawCache[il], rawStart, nRaw, kvStage, 0);
        if (mixed) {
            tl.launch(bb, tl.CPY_U16, ceilDiv((long) nComp * nHeadDim, 1024), 1, 1,
                    tl.args(tl.CPY_U16).i(nComp * nHeadDim),
                    Tl.B.b(kvStage, (long) nRaw * nHeadDim * 2), Tl.B.b(attnCompCache[il], 0));
        }
        Metal.Buffer mask = mixed
                ? cachedMask(false, pos0, n, nKeys, rawWindow, ratio, nComp)
                : cachedMask(true, pos0, n, nKeys, rawWindow, 0, 0);
        flashPrefill(bb, n, nKeys, scale, l_sinks(il), mask);
    }

    long l_sinks(int il) { return lw[il].attnSinks; }

    void stageF16(Metal.Batch bb, Metal.Buffer src, long srcOff, Scratch dstS, long dstOff, long n) {
        Metal.Buffer dst = dstS.get(metal, Math.max(1, dstOff + n * 2));
        tl.launch(bb, tl.CPY_F32_F16, ceilDiv(n, 1024), 1, 1,
                tl.args(tl.CPY_F32_F16).i(n),
                Tl.B.b(dst, dstOff), Tl.B.b(src, srcOff));
    }

    void stageRingF16(Metal.Batch bb, Metal.Buffer ring, int rawStart, int nRaw, Metal.Buffer dst, long dstOff) {
        int tail = Math.min(nRaw, rawCap - rawStart);
        tl.launch(bb, tl.CPY_F32_F16, ceilDiv((long) tail * nHeadDim, 1024), 1, 1,
                tl.args(tl.CPY_F32_F16).i(tail * nHeadDim),
                Tl.B.b(dst, dstOff), Tl.B.b(ring, (long) rawStart * nHeadDim * 4));
        if (tail < nRaw) {
            tl.launch(bb, tl.CPY_F32_F16, ceilDiv((long) (nRaw - tail) * nHeadDim, 1024), 1, 1,
                    tl.args(tl.CPY_F32_F16).i((nRaw - tail) * nHeadDim),
                    Tl.B.b(dst, dstOff + (long) tail * nHeadDim * 2), Tl.B.b(ring, 0));
        }
    }

    void indexerScores(Metal.Batch bb, int il, int pos0, int n, int nComp, float scale) {
        float indexScale = (float) (1.0 / Math.sqrt(idxHead * (double) idxDim));
        Metal.Buffer scores = pfIndexerScores.get(metal, (long) n * nComp * 4);
        tl.launch(bb, tl.INDEXER_SCORES, ceilDiv(nComp, 32), n, 1,
                tl.args(tl.INDEXER_SCORES).i(nComp).i(nComp).i(n).i(pos0).i(4).f(indexScale),
                Tl.B.b(idxCompCache[il], 0), Tl.B.b(pfIndexerQ, 0),
                Tl.B.b(scores, 0), Tl.B.b(pfIndexerW, 0));
    }

    void topkAndSort(Metal.Batch bb, int n, int nComp) {
        Metal.Buffer scores = pfIndexerScores.get(metal, (long) n * nComp * 4);
        Metal.Buffer topk = pfTopk.get(metal, (long) n * idxTopK * 4);
        tl.launch(bb, tl.TOPK_RADIX, n, 1, 1,
                tl.args(tl.TOPK_RADIX).i(n).i(idxTopK).i(nComp),
                Tl.B.b(topk, 0), Tl.B.b(scores, 0));
        Metal.Buffer sorted = pfTopkSorted.get(metal, (long) n * idxTopK * 4);
        tl.launch(bb, tl.SORT_I32_ASC_512, n, 1, 1,
                tl.args(tl.SORT_I32_ASC_512).i(n),
                Tl.B.b(sorted, 0), Tl.B.b(topk, 0));
    }

    void flashPrefill(Metal.Batch bb, int nQ, int nKeys, float scale, long sinksOff, Metal.Buffer mask) {
        tl.launch(bb, tl.FLASH_PREFILL, ceilDiv((long) nQ * nHead, 32), 1, 1,
                tl.args(tl.FLASH_PREFILL).i(nHead).i(nKeys).i(nQ * nHead).f(scale),
                Tl.B.b(pfHeads, 0), Tl.B.b(pfKvStage.get(metal, (long) nKeys * nHeadDim * 2), 0),
                Tl.B.b(mask, 0), Tl.B.b(pfQ, 0), mb(sinksOff));
    }

    /** Zero-prefix or nonzero chunk mask. Keyed per (pos0, nQ, nKeys, ratio,
     *  nComp); same-key layers share one fill, different parameters get a
     *  fresh buffer (never a refill of an encoded-against buffer). */
    Metal.Buffer cachedMask(boolean rawOnly, int pos0, int nQ, int nKeys, int window, int ratio, int nComp) {
        long key = (((long) pos0 << 34) | ((long) nQ << 22) | ((long) nKeys << 8)
                | ((long) (nComp & 0xff) << 1) | (rawOnly ? 1 : 0));
        Metal.Buffer hit = maskCache.get(key);
        if (hit != null) return hit;
        Metal.Buffer mask = metal.alloc((long) nQ * nKeys * 2);
        maskFillTarget = mask;
        if (rawOnly || nComp == 0) {
            if (pos0 == 0) fillRawPrefillMask(0, nQ, nKeys, window);
            else fillRawDecodeMask(nQ, nKeys, pos0, window);
        } else {
            if (pos0 == 0) fillStaticMixedMask(0, nQ, nQ, nComp, window, ratio);
            else fillMixedDecodeMask(nQ, nKeys - nComp, nComp, pos0, window, ratio);
        }
        maskCache.put(key, mask);
        return mask;
    }

    // mask fills (CPU, little-endian u16: 0 visible, 0xFC00 -inf)
    // (they fill the buffer most recently allocated by cachedMask)
    Metal.Buffer maskFillTarget;

    void fillRawPrefillMask(int qRow0, int nQ, int nKv, int window) {
        Metal.Buffer mask = maskFillTarget;
        for (int q = 0; q < nQ; q++) {
            int qpos = qRow0 + q;
            long row = (long) q * nKv * 2;
            for (int k = 0; k < nKv; k++) {
                boolean vis = k <= qpos && (window == 0 || qpos - k < window);
                mask.setShort(row + (long) k * 2, vis ? (short) 0 : (short) 0xFC00);
            }
        }
    }

    void fillStaticMixedMask(int qRow0, int nQ, int nTokens, int nComp, int window, int ratio) {
        Metal.Buffer mask = maskFillTarget;
        int nKeys = nTokens + nComp;
        for (int q = 0; q < nQ; q++) {
            int qpos = qRow0 + q;
            long row = (long) q * nKeys * 2;
            for (int k = 0; k < nTokens; k++) {
                boolean vis = k <= qpos && (window == 0 || qpos - k < window);
                mask.setShort(row + (long) k * 2, vis ? (short) 0 : (short) 0xFC00);
            }
            int nVisible = (qpos + 1) / ratio;
            for (int c = 0; c < nComp; c++) {
                mask.setShort(row + (long) (nTokens + c) * 2, c < nVisible ? (short) 0 : (short) 0xFC00);
            }
        }
    }

    void fillRawDecodeMask(int nQ, int nRaw, int pos0, int window) {
        Metal.Buffer mask = maskFillTarget;
        int lastPos = pos0 + nQ - 1;
        int firstRawPos = lastPos + 1 - nRaw;
        for (int q = 0; q < nQ; q++) {
            int qpos = pos0 + q;
            long row = (long) q * nRaw * 2;
            for (int k = 0; k < nRaw; k++) {
                int kpos = firstRawPos + k;
                boolean vis = kpos <= qpos && (window == 0 || qpos - kpos < window);
                mask.setShort(row + (long) k * 2, vis ? (short) 0 : (short) 0xFC00);
            }
        }
    }

    void fillMixedDecodeMask(int nQ, int nRaw, int nComp, int pos0, int window, int ratio) {
        Metal.Buffer mask = maskFillTarget;
        int nKeys = nRaw + nComp;
        int lastPos = pos0 + nQ - 1;
        int firstRawPos = lastPos + 1 - nRaw;
        for (int q = 0; q < nQ; q++) {
            int qpos = pos0 + q;
            long row = (long) q * nKeys * 2;
            for (int k = 0; k < nRaw; k++) {
                int kpos = firstRawPos + k;
                boolean vis = kpos <= qpos && (window == 0 || qpos - kpos < window);
                mask.setShort(row + (long) k * 2, vis ? (short) 0 : (short) 0xFC00);
            }
            int nVisible = (qpos + 1) / ratio;
            for (int c = 0; c < nComp; c++) {
                mask.setShort(row + (long) (nRaw + c) * 2, c < nVisible ? (short) 0 : (short) 0xFC00);
            }
        }
    }

    // ---- router (batch) ----

    void routerSelectBatch(Metal.Batch bb, int il, int n) {
        LW l = lw[il];
        if (l.hasTid2eid) {
            if (n == 1) {
                // decode: fused sqrt(softplus) + fused router weights (C n=1 path)
                tl.launch(bb, tl.UNARY_ROWS, ceilDiv(nExpert, 256), n, 1,
                        tl.args(tl.UNARY_ROWS).i(nExpert).i(OP_SOFTPLUS_SQRT).f(0).f(0).i(n).i(nExpert).i(nExpert),
                        Tl.B.b(pfRouterProbs, 0), Tl.B.b(pfRouterLogits, 0));
                tl.launch(bb, tl.GET_ROWS_I32, ceilDiv(nUsed, 32), n, 1,
                        tl.args(tl.GET_ROWS_I32).i(nUsed).i(n).i(nUsed).i(nUsed),
                        Tl.B.b(pfRouterSel, 0), Tl.B.b(pfTokens, 0), mb(l.tid2eid));
                tl.launch(bb, tl.ROUTER_WEIGHTS, n, 1, 1,
                        tl.args(tl.ROUTER_WEIGHTS).i(nExpert).i(n).f(EXPERT_SCALE),
                        Tl.B.b(pfRouterProbs, 0), Tl.B.b(pfRouterSel, 0), Tl.B.b(pfRouterW, 0));
                return;
            }
            // batch: C parity — separate softplus and sqrt kernels (intermediate
            // f32 rounding), then gather/sum/clamp/div/mul weight chain
            tl.launch(bb, tl.UNARY_ROWS, ceilDiv(nExpert, 256), n, 1,
                    tl.args(tl.UNARY_ROWS).i(nExpert).i(OP_SOFTPLUS).f(0).f(0).i(n).i(nExpert).i(nExpert),
                    Tl.B.b(pfRouterProbs, 0), Tl.B.b(pfRouterLogits, 0));
            tl.launch(bb, tl.UNARY_ROWS, ceilDiv(nExpert, 256), n, 1,
                    tl.args(tl.UNARY_ROWS).i(nExpert).i(OP_SQRT).f(0).f(0).i(n).i(nExpert).i(nExpert),
                    Tl.B.b(pfRouterProbs, 0), Tl.B.b(pfRouterProbs, 0));
            tl.launch(bb, tl.GET_ROWS_I32, ceilDiv(nUsed, 32), n, 1,
                    tl.args(tl.GET_ROWS_I32).i(nUsed).i(n).i(nUsed).i(nUsed),
                    Tl.B.b(pfRouterSel, 0), Tl.B.b(pfTokens, 0), mb(l.tid2eid));
            tl.launch(bb, tl.GATHER_SCALARS, n, 1, 1,
                    tl.args(tl.GATHER_SCALARS).i(nUsed).i(nUsed).i(n).i(nExpert).i(nUsed),
                    Tl.B.b(pfRouterW, 0), Tl.B.b(pfRouterSel, 0), Tl.B.b(pfRouterProbs, 0));
            tl.launch(bb, tl.SUM_ROWS, ceilDiv(n, 32), 1, 1,
                    tl.args(tl.SUM_ROWS).i(n).i(nUsed).i(nUsed),
                    Tl.B.b(pfRouterSum, 0), Tl.B.b(pfRouterW, 0));
            tl.launch(bb, tl.UNARY_ROWS, 1, n, 1,
                    tl.args(tl.UNARY_ROWS).i(1).i(OP_CLAMP).f(6.103515625e-5f).f(Float.POSITIVE_INFINITY).i(n).i(1).i(1),
                    Tl.B.b(pfRouterSum, 0), Tl.B.b(pfRouterSum, 0));
            tl.launch(bb, tl.BIN_ROWS, ceilDiv(nUsed, 256), n, 1,
                    tl.args(tl.BIN_ROWS).i(nUsed).i(1).i(n).i(1).i(nUsed).i(OP_DIV).i(n).i(nUsed),
                    Tl.B.b(pfRouterW, 0), Tl.B.b(pfRouterSum, 0), Tl.B.b(pfRouterW, 0));
            tl.launch(bb, tl.BIN_ROWS, ceilDiv(nUsed, 256), n, 1,
                    tl.args(tl.BIN_ROWS).i(nUsed).i(1).i(1).i(1).i(nUsed).i(OP_MUL).i(n).i(nUsed),
                    Tl.B.b(pfRouterW, 0), Tl.B.inline(Float.floatToRawIntBits(EXPERT_SCALE)), Tl.B.b(pfRouterW, 0));
            return;
        }
        // probs: n=1 keeps the fused dsv4_softplus_sqrt (decode parity);
        // n>1 uses the C batch chain — separate softplus and sqrt kernels
        if (n == 1) {
            tl.launch(bb, tl.UNARY_ROWS, ceilDiv(nExpert, 256), n, 1,
                    tl.args(tl.UNARY_ROWS).i(nExpert).i(OP_SOFTPLUS_SQRT).f(0).f(0).i(n).i(nExpert).i(nExpert),
                    Tl.B.b(pfRouterProbs, 0), Tl.B.b(pfRouterLogits, 0));
        } else {
            tl.launch(bb, tl.UNARY_ROWS, ceilDiv(nExpert, 256), n, 1,
                    tl.args(tl.UNARY_ROWS).i(nExpert).i(OP_SOFTPLUS).f(0).f(0).i(n).i(nExpert).i(nExpert),
                    Tl.B.b(pfRouterProbs, 0), Tl.B.b(pfRouterLogits, 0));
            tl.launch(bb, tl.UNARY_ROWS, ceilDiv(nExpert, 256), n, 1,
                    tl.args(tl.UNARY_ROWS).i(nExpert).i(OP_SQRT).f(0).f(0).i(n).i(nExpert).i(nExpert),
                    Tl.B.b(pfRouterProbs, 0), Tl.B.b(pfRouterProbs, 0));
        }
        // score for selection: probs + bias into the (now free) logits buffer,
        // keeping pfRouterProbs raw — weights are gathered from RAW probs (C parity)
        Metal.Buffer scoreB = pfRouterProbs;
        if (l.expProbsB != 0) {
            scoreB = pfRouterLogits;
            tl.launch(bb, tl.BIN_ROWS, ceilDiv(nExpert, 256), n, 1,
                    tl.args(tl.BIN_ROWS).i(nExpert).i(nExpert).i(1).i(nExpert).i(nExpert)
                            .i(OP_ADD).i(n).i(nExpert),
                    Tl.B.b(pfRouterProbs, 0), mb(l.expProbsB), Tl.B.b(scoreB, 0));
        }
        tl.launch(bb, tl.ARGSORT_DESC_TOPK_256, n, 1, 1,
                tl.args(tl.ARGSORT_DESC_TOPK_256).i(nUsed).i(n),
                Tl.B.b(scoreB, 0), Tl.B.b(pfRouterSel, 0));
        // weights = probs[sel]; sum; clamp; div; x1.5
        tl.launch(bb, tl.GATHER_SCALARS, n, 1, 1,
                tl.args(tl.GATHER_SCALARS).i(nUsed).i(nUsed).i(n).i(nExpert).i(nUsed),
                Tl.B.b(pfRouterW, 0), Tl.B.b(pfRouterSel, 0), Tl.B.b(pfRouterProbs, 0));
        tl.launch(bb, tl.SUM_ROWS, ceilDiv(n, 32), 1, 1,
                tl.args(tl.SUM_ROWS).i(n).i(nUsed).i(nUsed),
                Tl.B.b(pfRouterSum, 0), Tl.B.b(pfRouterW, 0));
        tl.launch(bb, tl.UNARY_ROWS, 1, n, 1,
                tl.args(tl.UNARY_ROWS).i(1).i(OP_CLAMP).f(6.103515625e-5f).f(Float.POSITIVE_INFINITY).i(n).i(1).i(1),
                Tl.B.b(pfRouterSum, 0), Tl.B.b(pfRouterSum, 0));
        tl.launch(bb, tl.BIN_ROWS, ceilDiv(nUsed, 256), n, 1,
                tl.args(tl.BIN_ROWS).i(nUsed).i(1).i(n).i(1).i(nUsed).i(OP_DIV).i(n).i(nUsed),
                Tl.B.b(pfRouterW, 0), Tl.B.b(pfRouterSum, 0), Tl.B.b(pfRouterW, 0));
        tl.launch(bb, tl.BIN_ROWS, ceilDiv(nUsed, 256), n, 1,
                tl.args(tl.BIN_ROWS).i(nUsed).i(1).i(1).i(1).i(nUsed).i(OP_MUL).i(n).i(nUsed),
                Tl.B.b(pfRouterW, 0), Tl.B.inline(Float.floatToRawIntBits(EXPERT_SCALE)), Tl.B.b(pfRouterW, 0));
    }

    // ---- routed MoE (batch kernels for n<32, packed path for larger chunks) ----

    // C parity: use_packed_mpp requires n_tokens >= 32 (TileLang MoE enabled)
    static final int PACKED_MOE_MIN = Integer.getInteger("ds4j.packedMoeMin", 32);

    void moeBatch(Metal.Batch bb, LW l, int n) {
        if (n < PACKED_MOE_MIN) {
            tl.launch(bb, tl.MOE_GATE_UP_BATCH, ffMid / 4, nUsed, n,
                    tl.args(tl.MOE_GATE_UP_BATCH).f(SWIGLU_CLAMP).i(n).i(nUsed).i(ffMid),
                    Tl.B.b(pfRoutedGate, 0), Tl.B.b(iq2Tables, 0), Tl.B.b(pfRouterSel, 0),
                    Tl.B.b(iq2Tables, 8192), Tl.B.b(pfRoutedMid, 0), Tl.B.b(pfRouterW, 0),
                    Tl.B.b(pfRoutedUp, 0), mb(l.ffnGateExps), mb(l.ffnUpExps), Tl.B.b(pfFfnNorm, 0));
            tl.launch(bb, tl.MOE_DOWN_BATCH, 4096 / 4, n, 1,
                    tl.args(tl.MOE_DOWN_BATCH).i(n).i(nUsed).i(4096),
                    Tl.B.b(pfRouterSel, 0), Tl.B.b(pfRoutedMid, 0), Tl.B.b(pfRoutedOut, 0),
                    mb(l.ffnDownExps));
            return;
        }
        moePacked(bb, l, n);
    }

    /** Debug driver: batch vs packed MoE on the current layer-0 prefilled state. */
    public void moeXcheck(int layerIdx, int n) {
        LW l = lw[layerIdx];
        int len = n * 4096;
        float[] batchRes = new float[len];
        try (Metal.Batch b1 = metal.begin()) {
            moeBatchPlain(b1, l, n);
            b1.commit(); b1.waitGpu();
        }
        for (int i = 0; i < len; i++) batchRes[i] = pfRoutedOut.getFloat((long) i * 4);
        float[] packedRes = new float[len];
        try (Metal.Batch b2 = metal.begin()) {
            moePacked(b2, l, n);
            b2.commit(); b2.waitGpu();
        }
        for (int i = 0; i < len; i++) packedRes[i] = pfRoutedOut.getFloat((long) i * 4);
        float md = 0; int at = -1, nDiff = 0;
        for (int i = 0; i < len; i++) {
            float d = Math.abs(batchRes[i] - packedRes[i]);
            if (d > 0.001f) nDiff++;
            if (d > md) { md = d; at = i; }
        }
        System.out.printf("MOE XCHECK layer %d n=%d: batch-vs-packed max diff %.5f at tok %d dim %d (%d/%d elems >1e-3)%n",
                layerIdx, n, md, at / 4096, at % 4096, nDiff, len);
    }

    /** Stage bisect: compare batch gate/up vs packed gate/up (before swiglu). */
    public void moeXcheckStage(int layerIdx, int n) {
        LW l = lw[layerIdx];
        int rows = n * nUsed, len = rows * ffMid;
        float[] batchGate = new float[len], batchUp = new float[len];
        try (Metal.Batch b1 = metal.begin()) {
            tl.launch(b1, tl.MOE_GATE_UP_BATCH, ffMid / 4, nUsed, n,
                    tl.args(tl.MOE_GATE_UP_BATCH).f(SWIGLU_CLAMP).i(n).i(nUsed).i(ffMid),
                    Tl.B.b(pfRoutedGate, 0), Tl.B.b(iq2Tables, 0), Tl.B.b(pfRouterSel, 0),
                    Tl.B.b(iq2Tables, 8192), Tl.B.b(pfRoutedMid, 0), Tl.B.b(pfRouterW, 0),
                    Tl.B.b(pfRoutedUp, 0), mb(l.ffnGateExps), mb(l.ffnUpExps), Tl.B.b(pfFfnNorm, 0));
            b1.commit(); b1.waitGpu();
        }
        for (int i = 0; i < len; i++) { batchGate[i] = pfRoutedGate.getFloat((long) i * 4); batchUp[i] = pfRoutedUp.getFloat((long) i * 4); }
        // packed: map0 + offsets + pack rhs1 + two gemms
        long pairRows = (long) nUsed * n;
        long workCap = (pairRows + 31L * nExpert + 31) / 32;
        long countsOff = 0;
        long hidsOff = nExpert * 4L;
        long workOff = (hidsOff + (long) nExpert * n * 4 + 7) & ~7;
        Metal.Buffer idMap = pfMoeMap.get(metal, workOff + 8 + workCap * 8);
        try (Metal.Batch b2 = metal.begin()) {
            tl.launch(b2, tl.MOE_MAP0, 1, 1, 1,
                    tl.args(tl.MOE_MAP0).i(nExpert).i(n),
                    Tl.B.b(idMap, countsOff), Tl.B.b(idMap, hidsOff),
                    Tl.B.b(pfRouterSel, 0), Tl.B.b(idMap, workOff));
            long header = ((nExpert + 31) / 32) * 32 * 4L;
            Metal.Buffer rhs1 = pfPackedRhs.get(metal, header + (pairRows + 32) * 4096 * 2);
            moePackRhs(b2, false, pfFfnNorm, 0, 0, (long) nEmb * 4, idMap, countsOff, hidsOff, workOff,
                    rhs1, header, 4096, n, workCap);
            moePackedGemm(b2, true, pfRoutedGate, l.ffnGateExps, rhs1, header, idMap, countsOff, hidsOff, workOff,
                    4096, ffMid, n, workCap, pairRows);
            moePackedGemm(b2, true, pfRoutedUp, l.ffnUpExps, rhs1, header, idMap, countsOff, hidsOff, workOff,
                    4096, ffMid, n, workCap, pairRows);
            b2.commit(); b2.waitGpu();
        }
        // also read back counts/hids/work and the offsets header for inspection
        int[] counts = new int[nExpert];
        for (int i = 0; i < nExpert; i++) counts[i] = idMap.getInt(countsOff + (long) i * 4);
        int total = 0; for (int c : counts) total += c;
        float mdG = 0, mdU = 0; int atG = -1, nDiffG = 0;
        for (int i = 0; i < len; i++) {
            float dg = Math.abs(batchGate[i] - pfRoutedGate.getFloat((long) i * 4));
            float du = Math.abs(batchUp[i] - pfRoutedUp.getFloat((long) i * 4));
            if (dg > 0.001f) nDiffG++;
            if (dg > mdG) { mdG = dg; atG = i; }
            if (du > mdU) mdU = du;
        }
        System.out.printf("MOE STAGE XCHECK layer %d n=%d: gate max diff %.5f at pair %d dim %d (%d/%d>1e-3), up max diff %.5f, pair rows total=%d (expect %d)%n",
                layerIdx, n, mdG, atG / ffMid, atG % ffMid, nDiffG, len, mdU, total, n * nUsed);
        // which pair rows are bad, and which expert do they route to?
        java.util.TreeSet<Integer> badPairs = new java.util.TreeSet<>();
        for (int p = 0; p < rows; p++)
            for (int d = 0; d < ffMid; d++)
                if (Math.abs(batchGate[p * ffMid + d] - pfRoutedGate.getFloat(((long) p * ffMid + d) * 4)) > 0.001f) {
                    badPairs.add(p); break;
                }
        System.out.printf("  bad gate pairs: %d of %d: %s%n", badPairs.size(), rows,
                badPairs.size() <= 20 ? badPairs.toString() : new java.util.ArrayList<>(badPairs).subList(0, 20).toString());
        for (int p : badPairs) {
            int tok = p / nUsed, slot = p % nUsed;
            int expert = pfRouterSel.getInt(((long) tok * nUsed + slot) * 4);
            System.out.printf("  pair %d (tok %d slot %d expert %d): batch[0..3]=%f %f %f packed=%f %f %f count=%d%n",
                    p, tok, slot, expert,
                    batchGate[p * ffMid], batchGate[p * ffMid + 1], batchGate[p * ffMid + 2],
                    pfRoutedGate.getFloat((long) p * ffMid * 4), pfRoutedGate.getFloat((long) (p * ffMid + 1) * 4),
                    pfRoutedGate.getFloat((long) (p * ffMid + 2) * 4), counts[expert]);
        }
    }

    /** the original small-n path, callable standalone for cross-checks */
    void moeBatchPlain(Metal.Batch bb, LW l, int n) {
        tl.launch(bb, tl.MOE_GATE_UP_BATCH, ffMid / 4, nUsed, n,
                tl.args(tl.MOE_GATE_UP_BATCH).f(SWIGLU_CLAMP).i(n).i(nUsed).i(ffMid),
                Tl.B.b(pfRoutedGate, 0), Tl.B.b(iq2Tables, 0), Tl.B.b(pfRouterSel, 0),
                Tl.B.b(iq2Tables, 8192), Tl.B.b(pfRoutedMid, 0), Tl.B.b(pfRouterW, 0),
                Tl.B.b(pfRoutedUp, 0), mb(l.ffnGateExps), mb(l.ffnUpExps), Tl.B.b(pfFfnNorm, 0));
        tl.launch(bb, tl.MOE_DOWN_BATCH, 4096 / 4, n, 1,
                tl.args(tl.MOE_DOWN_BATCH).i(n).i(nUsed).i(4096),
                Tl.B.b(pfRouterSel, 0), Tl.B.b(pfRoutedMid, 0), Tl.B.b(pfRoutedOut, 0),
                mb(l.ffnDownExps));
    }

    void moePacked(Metal.Batch bb, LW l, int n) {
        long pairRows = (long) nUsed * n;
        long workCap = (pairRows + 31L * nExpert + 31) / 32;
        long countsOff = 0;                       // counts[256] u32
        long hidsOff = nExpert * 4L;             // hids[256*n] i32 (expert-major, C layout)
        long workOff = (hidsOff + (long) nExpert * n * 4 + 7) & ~7;
        Metal.Buffer idMap = pfMoeMap.get(metal, workOff + 8 + workCap * 8);
        // map0
        tl.launch(bb, tl.MOE_MAP0, 1, 1, 1,
                tl.args(tl.MOE_MAP0).i(nExpert).i(n),
                Tl.B.b(idMap, countsOff), Tl.B.b(idMap, hidsOff),
                Tl.B.b(pfRouterSel, 0), Tl.B.b(idMap, workOff));
        // pack x (f32 source, one row per token broadcast over slots)
        long header = ((nExpert + 31) / 32) * 32 * 4L;
        Metal.Buffer rhs1 = pfPackedRhs.get(metal, header + (pairRows + 32) * 4096 * 2);
        moePackRhs(bb, false, pfFfnNorm, 0, 0, (long) nEmb * 4, idMap, countsOff, hidsOff, workOff,
                rhs1, header, 4096, n, workCap);
        // gate + up packed gemms
        moePackedGemm(bb, true, pfRoutedGate, l.ffnGateExps, rhs1, header, idMap, countsOff, hidsOff, workOff,
                4096, ffMid, n, workCap, pairRows);
        moePackedGemm(bb, true, pfRoutedUp, l.ffnUpExps, rhs1, header, idMap, countsOff, hidsOff, workOff,
                4096, ffMid, n, workCap, pairRows);
        // activation with route weights -> f16 mid
        tl.launch(bb, tl.MOE_SWIGLU_WEIGHT_F16, 8, n * nUsed, 1,
                tl.args(tl.MOE_SWIGLU_WEIGHT_F16).f(SWIGLU_CLAMP).i(n * nUsed),
                Tl.B.b(pfRoutedGate, 0), Tl.B.b(pfRoutedMid16, 0),
                Tl.B.b(pfRoutedUp, 0), Tl.B.b(pfRouterW, 0));
        // pack mid (f16 source, per-slot rows)
        Metal.Buffer rhs2 = pfMoeRhs2.get(metal, header + (pairRows + 32) * 2048 * 2);
        moePackRhs(bb, true, pfRoutedMid16, 0, (long) ffMid * 2, (long) nUsed * ffMid * 2,
                idMap, countsOff, hidsOff, workOff, rhs2, header, 2048, n, workCap);
        moePackedGemm(bb, false, pfRoutedDown, l.ffnDownExps, rhs2, header, idMap, countsOff, hidsOff, workOff,
                2048, 4096, n, workCap, pairRows);
        tl.launch(bb, tl.MOE_SUM6, 16, n, 1,
                tl.args(tl.MOE_SUM6).i(n),
                Tl.B.b(pfRoutedOut, 0), Tl.B.b(pfRoutedDown, 0));
    }

    void moePackRhs(Metal.Batch bb, boolean srcF16, Metal.Buffer src, long srcOff,
                    long slotStrideBytes, long tokStrideBytes,
                    Metal.Buffer idMap, long countsOff, long hidsOff, long workOff,
                    Metal.Buffer rhs, long header, int kDim, int n, long workCap) {
        tl.launch(bb, tl.MOE_PACKED_OFFSETS, 1, 1, 1,
                tl.args(tl.MOE_PACKED_OFFSETS).i(nExpert),
                Tl.B.b(idMap, countsOff), Tl.B.b(rhs, 0));
        int esz = srcF16 ? 2 : 4;
        Tl.Spec pack = srcF16 ? tl.MOE_PACK_RHS_F16 : tl.MOE_PACK_RHS_F32;
        tl.launch(bb, pack, (int) Math.min(Integer.MAX_VALUE, workCap * 32), 1, 1,
                tl.args(pack).i(kDim).i(n).i(nUsed)
                        .i(slotStrideBytes / (4L * esz)).i(tokStrideBytes / (4L * esz)).i(workCap),
                Tl.B.b(idMap, countsOff), Tl.B.b(rhs, header), Tl.B.b(idMap, hidsOff),
                Tl.B.b(rhs, 0), Tl.B.b(src, srcOff), Tl.B.b(idMap, workOff));
    }

    void moePackedGemm(Metal.Batch bb, boolean iq2, Metal.Buffer dst, long wOff,
                       Metal.Buffer rhs, long header, Metal.Buffer idMap,
                       long countsOff, long hidsOff, long workOff,
                       int inDim, int outDim, int n, long workCap, long pairRows) {
        long rhsRows = pairRows + 32;
        if (iq2) {
            tl.launch(bb, tl.MOE_PACKED_GEMM_IQ2, (int) workCap, outDim / 64, 1,
                    tl.args(tl.MOE_PACKED_GEMM_IQ2).i(n).i(outDim).i(rhsRows).i(workCap),
                    Tl.B.b(idMap, countsOff), Tl.B.b(dst, 0), Tl.B.b(iq2Tables, 8320),
                    Tl.B.b(idMap, hidsOff), Tl.B.b(iq2Tables, 8192), Tl.B.b(rhs, 0),
                    Tl.B.b(rhs, header), mb(wOff), Tl.B.b(idMap, workOff));
        } else {
            tl.launch(bb, tl.MOE_PACKED_GEMM_Q2K, (int) workCap, outDim / 64, 1,
                    tl.args(tl.MOE_PACKED_GEMM_Q2K).i(n).i(outDim).i(rhsRows).i(workCap),
                    Tl.B.b(idMap, countsOff), Tl.B.b(dst, 0), Tl.B.b(idMap, hidsOff),
                    Tl.B.b(rhs, 0), Tl.B.b(rhs, header), mb(wOff), Tl.B.b(idMap, workOff));
        }
    }

}
