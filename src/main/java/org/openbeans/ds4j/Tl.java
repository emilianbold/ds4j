package org.openbeans.ds4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * TileLang kernel registry and generic launcher. Mirrors the C dispatch
 * layer (ds4_tl_metal.h + gen/metal/tl_launch_meta.h): buffers bind in the
 * generated kernel's declared order, scalars travel as the packed
 * <name>_args_t struct (each field an 8-byte slot: value, pad) via setBytes
 * right after the last buffer, grids come from the generated _grid() rules,
 * and autotune variants are selected per tune group from the tuned JSON.
 */
public final class Tl {
    public final Metal metal;

    public static final String[] TUNE_GROUPS = {
            "gemm", "q8_gemv_f32", "moe_down", "moe_gate_up", "f16_gemv",
            "f16_gemv_batch", "hc_mix_gemm", "hc_mix", "quad",
            "flash_decode", "indexed_attn", "flash_prefill"};

    private final int[] tuneSel = new int[TUNE_GROUPS.length];

    /** One autotune variant: MSL symbol + threads per threadgroup. */
    public record Var(String symbol, int tx) {}

    /**
     * Kernel spec. args: names in struct order; isFloat: encode as f32.
     * Variants: for gemv-like kernels, grid.x = primaryDim / div(variant).
     */
    public static final class Spec {
        public final Var[] vars;
        public final int tuneId;
        public final int nBufs;
        public final String[] args;
        public final boolean[] isFloat;
        final int[] gridDiv; // per variant: divisor for the primary grid dim (0 = fixed grid)
        Spec(Var[] vars, int tuneId, int nBufs, String[] args, boolean[] isFloat, int[] gridDiv) {
            this.vars = vars; this.tuneId = tuneId; this.nBufs = nBufs;
            this.args = args; this.isFloat = isFloat; this.gridDiv = gridDiv;
        }
    }

    /** A buffer binding: Metal buffer handle + byte offset (handle 0 = inline i32 in off). */
    public record B(long handle, long off) {
        public static B b(Metal.Buffer buf, long off) { return new B(buf.handle, off); }
        public static B inline(int value) { return new B(0, value); }
        public long bufHandle() { return handle; }
    }

    /** Scalar-args packer for one launch: <name>_args_t fields, each an
     * 8-byte slot (value in the low 4 bytes, zero padding), little endian.
     * Slices come from a bump-allocated process-wide segment (args are
     * consumed synchronously inside launch(), so a wrapping cursor is safe
     * and nothing is allocated per launch). */
    public final class Args {
        private final MemorySegment mem;
        private int i;
        Args(int n) {
            long need = Math.max(1, n) * 8L;
            if (argCursor + need > ARG_SCRATCH.byteSize()) argCursor = 0;
            mem = ARG_SCRATCH.asSlice(argCursor, need).fill((byte) 0);
            argCursor += need;
        }
        public Args f(double v) {
            mem.set(ValueLayout.JAVA_INT, (long) (i++) * 8, Float.floatToRawIntBits((float) v));
            return this;
        }
        public Args i(long v) {
            mem.set(ValueLayout.JAVA_INT, (long) (i++) * 8, (int) v);
            return this;
        }
    }

    private static final MemorySegment ARG_SCRATCH = Arena.global().allocate(8192);
    private static long argCursor = 0;
    private static final MemorySegment SCRATCH4 = Arena.global().allocate(4);

    public Tl(Metal metal, String tunedDir) {
        this.metal = metal;
        // gpu key: lowercase, non-alnum runs -> '-', like ds4_tl_gpu_key()
        String key = metal.deviceName.toLowerCase().replaceAll("[^a-z0-9]+", "-");
        String text = null;
        try {
            if (tunedDir != null && !tunedDir.isBlank()) {
                Path f = Path.of(tunedDir, key + ".json");
                if (Files.exists(f)) text = Files.readString(f);
            }
        } catch (Exception ignore) {}
        if (text == null)
            text = MetalSources.tunedJson(key); // packaged jar resource
        if (text != null) {
            // the C scanner reads every "key": "value" pair; group names map
            // to suffix indices; everything else is ignored
            Matcher m = Pattern.compile("\"([^\"]+)\"\\s*:\\s*\"([^\"]+)\"").matcher(text);
            while (m.find()) {
                String k = m.group(1), v = m.group(2);
                for (int g = 0; g < TUNE_GROUPS.length; g++) {
                    if (TUNE_GROUPS[g].equals(k)) {
                        String[] sfx = groupSuffixes(g);
                        for (int vi = 0; vi < sfx.length; vi++)
                            if (sfx[vi].equals(v)) tuneSel[g] = vi;
                    }
                }
            }
        }
    }

    static String[] groupSuffixes(int g) {
        return switch (g) {
            case 0 -> new String[]{"default", "k64", "t128", "k64t128"};
            case 1 -> S_GEMV;
            case 2, 3 -> S_MOE;
            case 4 -> S_F16;
            case 5 -> new String[]{"default", "t2", "t4"};
            case 6 -> new String[]{"default", "k32", "k128", "m32"};
            case 7 -> S_MIX;
            case 8 -> S_QUAD;
            case 9, 10 -> S_K128;
            case 11 -> new String[]{"default", "n64", "k128"};
            default -> new String[]{"default"};
        };
    }

    public Args args(Spec s) { return new Args(s.args.length); }

    public int sel(Spec s) {
        if (s.tuneId < 0) return 0;
        int v = tuneSel[s.tuneId];
        if (v < 0 || v >= s.vars.length) return 0;
        return v;
    }

    public int tx(Spec s) { return s.vars[sel(s)].tx(); }

    public int div(Spec s) {
        int v = sel(s);
        return s.gridDiv.length > v ? s.gridDiv[v] : 0;
    }

    // ---- launcher ----

    /** Perf diagnostics: count of kernel launches (per process). */
    public static final java.util.concurrent.atomic.AtomicLong LAUNCHES =
            new java.util.concurrent.atomic.AtomicLong();
    static final boolean TRACE = Boolean.getBoolean("ds4j.trace");
    /** Symbol histogram for perf parity analysis (reset between phases). */
    public static final java.util.Map<String, long[]> TRACE_COUNTS =
            java.util.Collections.synchronizedMap(new java.util.LinkedHashMap<>());

    public void launch(Metal.Batch b, Spec s, long gx, long gy, long gz, Args a, B... bufs) {
        LAUNCHES.incrementAndGet();
        String sym = s.vars[sel(s)].symbol();
        if (TRACE) {
            System.err.println("tl: " + sym + " tg=" + gx + "x" + gy + "x" + gz);
            TRACE_COUNTS.computeIfAbsent(sym, k -> new long[1])[0]++;
        }
        if (bufs.length != s.nBufs)
            throw new IllegalArgumentException(sym + ": expected " + s.nBufs + " buffers, got " + bufs.length);
        Var v = s.vars[sel(s)];
        for (int i = 0; i < bufs.length; i++) {
            if (bufs[i].handle == 0) {
                SCRATCH4.set(ValueLayout.JAVA_INT, 0, (int) bufs[i].off);
                Bridge.setBytes(b.enc, SCRATCH4.address(), 4, i);
            } else {
                Bridge.setBuffer(b.enc, bufs[i].handle, bufs[i].off, i);
            }
        }
        if (s.args.length > 0) Bridge.setBytes(b.enc, a.mem.address(), (long) s.args.length * 8, s.nBufs);
        else Bridge.setBytes(b.enc, SCRATCH4.address(), 4, s.nBufs);
        Bridge.setPipeline(b.enc, metal.pipeline(v.symbol()));
        Bridge.dispatchThreadgroups(b.enc, (int) gx, (int) gy, (int) gz, v.tx(), 1, 1);
    }

    // ---- spec builders ----
    private static String sym(String base, String suffix) {
        return suffix == null || suffix.equals("default")
                ? base + "_kernel" : base + "__" + suffix + "_kernel";
    }

    static Spec fixed(String base, int tx, int nBufs, String[] args, boolean[] isFloat) {
        return new Spec(new Var[]{new Var(base + "_kernel", tx)}, -1, nBufs, args, isFloat, new int[]{0});
    }

    static Spec tuned(String base, int tuneId, int[] txs, String[] suffixes, int nBufs, String[] args, boolean[] isFloat, int... divs) {
        Var[] vars = new Var[txs.length];
        for (int i = 0; i < txs.length; i++) vars[i] = new Var(sym(base, suffixes[i]), txs[i]);
        return new Spec(vars, tuneId, nBufs, args, isFloat, divs);
    }

    // =====================================================================
    // Kernel specs used by the decode path. Names/args mirror tl_launch_meta.h.
    // =====================================================================

    private static final String[] A_EPS_ROWS = {"eps", "rows"};
    private static final boolean[] F_EPS_ROWS = {true, false};

    public final Spec RMS_NORM_W4096  = fixed("tl_rms_norm_weight_f32_4096", 1024, 3, A_EPS_ROWS, F_EPS_ROWS);
    public final Spec RMS_NORM_W512   = fixed("tl_rms_norm_weight_f32_512", 128, 3, A_EPS_ROWS, F_EPS_ROWS);
    public final Spec RMS_NORM_W128  = fixed("tl_rms_norm_weight_f32_128", 32, 3, A_EPS_ROWS, F_EPS_ROWS);
    public final Spec RMS_NORM_PLAIN16384 = fixed("tl_rms_norm_plain_f32_16384", 1024, 2, A_EPS_ROWS, F_EPS_ROWS);

    public final Spec GET_ROWS_F16_F32 = fixed("tl_get_rows_f16_f32", 256, 3,
            new String[]{"dst_stride4", "rows", "table_stride4", "width"},
            new boolean[]{false, false, false, false});
    public final Spec GET_ROWS_I32 = fixed("tl_get_rows_i32", 32, 3,
            new String[]{"dst_stride", "rows", "table_stride", "width"},
            new boolean[]{false, false, false, false});

    public final Spec REPEAT_ROWS = fixed("tl_repeat_rows_f32", 256, 2,
            new String[]{"dst_rep_stride4", "dst_row_stride4", "reps", "rows", "src_stride4", "width"},
            new boolean[]{false, false, false, false, false, false});

    // tune group variants
    static final String[] S_GEMV = {"default", "r2", "r8", "t2"};      // q8_gemv_f32 (group 1)
    static final String[] S_F16  = {"default", "r1", "r2", "t2"};      // f16_gemv (group 4)
    static final String[] S_MOE  = {"default", "r8", "r8g4", "r2"};    // moe groups 2/3
    static final String[] S_QUAD = {"default", "r2", "r8"};            // quad (group 8)
    static final String[] S_MIX  = {"default", "r2", "t512", "r2t512"}; // hc_mix (group 7)
    static final String[] S_K128 = {"default", "k128"};                 // groups 9/10
    static final String[] S_GEMMV = {"default", "k64", "t128", "k64t128"}; // gemm (group 0)

    public final Spec HC_MIX16384 = tuned("tl_hc_rms_norm_mix_f16_16384", 7,
            new int[]{256, 256, 512, 512}, S_MIX, 3,
            new String[]{"eps", "out_dim"}, new boolean[]{true, false}, 1, 2, 1, 2);

    public final Spec HC_SPLIT_NORM4 = fixed("tl_hc_split_weighted_sum_norm4_f32_4096", 1024, 8,
            new String[]{"eps", "norm_eps", "rows", "sinkhorn_iters"},
            new boolean[]{true, true, false, false});
    /** Unfused n==1 batch path: split + pre-gated collapse, no norm. */
    public final Spec HC_SPLIT4 = fixed("tl_hc_split_weighted_sum4_f32_4096", 1024, 6,
            new String[]{"eps", "rows", "sinkhorn_iters"},
            new boolean[]{true, false, false});

    public final Spec QKV_QUAD_APE16 = tuned("tl_qkv_pair_quad_compressor_store_f32_4096_ape16", 8,
            new int[]{128, 64, 256}, S_QUAD, 19,
            new String[]{"kv_dim", "pos", "q_dim", "ratio", "width0", "width1"},
            new boolean[]{false, false, false, false, false, false}, 4, 2, 8);
    public final Spec QKV_QUAD_APE32 = tuned("tl_qkv_pair_quad_compressor_store_f32_4096_ape32", 8,
            new int[]{128, 64, 256}, S_QUAD, 19,
            new String[]{"kv_dim", "pos", "q_dim", "ratio", "width0", "width1"},
            new boolean[]{false, false, false, false, false, false}, 4, 2, 8);

    public final Spec Q8_GEMV_4096 = tuned("tl_q8_0_gemv_f32_4096", 1, new int[]{128, 64, 256, 256}, S_GEMV, 3,
            new String[]{"out_dim"}, new boolean[]{false}, 4, 2, 8, 4);
    public final Spec Q8_GEMV_1024 = tuned("tl_q8_0_gemv_f32_1024", 1, new int[]{128, 64, 256, 256}, S_GEMV, 3,
            new String[]{"out_dim"}, new boolean[]{false}, 4, 2, 8, 4);
    public final Spec Q8_GEMV_8192 = tuned("tl_q8_0_gemv_f32_8192", 1, new int[]{256, 128, 512, 512}, S_GEMV, 3,
            new String[]{"out_dim"}, new boolean[]{false}, 4, 2, 8, 4);

    public final Spec QKV_RMSNORM_KV_ROPE_FP8_STORE = fixed(
            "tl_qkv_rms_norm_kv_rope_fp8_store_f32_1024_512", 256, 7,
            new String[]{"attn_factor", "beta_fast", "beta_slow", "eps", "ext_factor",
                        "freq_base", "freq_scale", "n_ctx_orig", "pos0", "raw_row"},
            new boolean[]{true, true, true, true, true, true, true, false, false, false});

    public final Spec HEAD_RMSNORM_ROPE = fixed("tl_head_rms_norm_rope_tail_f32_512_64", 64, 1,
            new String[]{"attn_factor", "beta_fast", "beta_slow", "eps", "ext_factor",
                        "freq_base", "freq_scale", "inverse", "n_ctx_orig", "n_head", "n_tok", "pos0"},
            new boolean[]{true, true, true, true, true, true, true, false, false, false, false, false});

    public final Spec COMPRESSOR_SHIFT4 = fixed("tl_compressor_shift_ratio4_f32", 256, 2,
            new String[]{"n4"}, new boolean[]{false});
    public final Spec POOL_RATIO4_512 = fixed("tl_compressor_pool_ratio4_f32_512", 128, 3,
            new String[]{}, new boolean[]{});
    public final Spec POOL_RATIO4_128 = fixed("tl_compressor_pool_ratio4_f32_128", 128, 3,
            new String[]{}, new boolean[]{});

    public final Spec COMP_ROW_FINALIZE = fixed("tl_comp_row_finalize_f32_512_128", 256, 9,
            new String[]{"attn_factor", "beta_fast", "beta_slow", "ext_factor", "freq_base",
                        "freq_scale", "n_ctx_orig", "pos0", "rms_eps"},
            new boolean[]{true, true, true, true, true, true, false, false, true});

    public final Spec SOFTMAX_POOL = fixed("tl_softmax_pool_f32", 256, 3,
            new String[]{"d1", "head_dim", "kv_s0", "kv_s1", "kv_s2", "n_comp", "pooled",
                        "sc_s0", "sc_s1", "sc_s2"},
            new boolean[]{false, false, false, false, false, false, false, false, false, false});

    public final Spec ROPE_TAIL_512_H1 = fixed("tl_rope_tail_f32_512_64_h1", 32, 1,
            A_ROPE, F_ROPE);
    public final Spec ROPE_TAIL_512_H64 = fixed("tl_rope_tail_f32_512_64_h64", 256, 1,
            A_ROPE, F_ROPE);
    public final Spec ROPE_TAIL_128_H64 = fixed("tl_rope_tail_f32_128_64_h64", 256, 1,
            A_ROPE, F_ROPE);
    public final Spec ROPE_TAIL_128_H1 = fixed("tl_rope_tail_f32_128_64_h1", 32, 1,
            A_ROPE, F_ROPE);

    static final String[] A_ROPE = {"attn_factor", "beta_fast", "beta_slow", "ext_factor",
                                    "freq_base", "freq_scale", "inverse", "n_ctx_orig", "n_tok", "pos0", "pos_step"};
    static final boolean[] F_ROPE = {true, true, true, true, true, true, false, false, false, false, false};

    public final Spec FP8_QUANTIZE = fixed("tl_fp8_kv_quantize_f32_512_64", 64, 2,
            new String[]{"rows"}, new boolean[]{false});

    public final Spec CPY_F32_F16 = fixed("tl_cpy_f32_f16", 256, 2,
            new String[]{"n"}, new boolean[]{false});

    public final Spec FLASH_KV_STAGE = fixed("tl_flash_kv_stage_f16_512", 256, 3,
            new String[]{"n_comp", "n_raw", "raw_cap", "raw_start"},
            new boolean[]{false, false, false, false});

    public final Spec FLASH_PARTIAL = tuned("tl_flash_decode_partial_f16_512", 9,
            new int[]{256, 256}, S_K128, 4,
            new String[]{"n_head", "n_keys", "scale"}, new boolean[]{false, false, true}, 0, 0);

    public final Spec FLASH_REDUCE = fixed("tl_flash_decode_reduce_f16_512", 128, 4,
            new String[]{"attn_factor", "beta_fast", "beta_slow", "do_rope", "ext_factor",
                        "freq_base", "freq_scale", "n_ctx_orig", "n_head", "nsplit", "pos0"},
            new boolean[]{true, true, true, false, true, true, true, false, false, false, false});

    public final Spec INDEXED_ATTN = tuned("tl_indexed_attention_f16_512_comp16", 10,
            new int[]{256, 256}, S_K128, 8,
            new String[]{"n_comp", "n_head", "n_raw", "n_splits", "n_tokens", "pos0",
                        "ratio", "raw_cap", "raw_start", "scale", "sorted_topk", "top_k", "window"},
            new boolean[]{false, false, false, false, false, false, false, false, false, true, false, false, false}, 0, 0);

    public final Spec F16_GEMV_4096 = tuned("tl_f16_gemv_f32_4096", 4, new int[]{128, 32, 64, 256}, S_F16, 3,
            new String[]{"out_dim"}, new boolean[]{false}, 4, 1, 2, 4);
    public final Spec F16_GEMV_1024 = tuned("tl_f16_gemv_f32_1024", 4, new int[]{128, 32, 64, 256}, S_F16, 3,
            new String[]{"out_dim"}, new boolean[]{false}, 4, 1, 2, 4);
    public final Spec F16_GEMV_16384 = tuned("tl_f16_gemv_f32_16384", 4, new int[]{256, 128, 256, 512}, S_F16, 3,
            new String[]{"out_dim"}, new boolean[]{false}, 1, 1, 2, 1);

    public final Spec ROUTER_FINALIZE = fixed("tl_router_finalize_f32_256_6", 256, 5,
            new String[]{"has_bias"}, new boolean[]{false});

    public final Spec UNARY_ROWS = fixed("tl_unary_rows_f32", 256, 2,
            new String[]{"dst_stride", "op", "p0", "p1", "rows", "src_stride", "width"},
            new boolean[]{false, false, true, true, false, false, false});

    public final Spec ROUTER_WEIGHTS = fixed("tl_router_weights_f32_6", 32, 3,
            new String[]{"n_expert", "rows", "scale"}, new boolean[]{false, false, true});

    public final Spec ADD2 = fixed("tl_add2_f32", 256, 3,
            new String[]{"n"}, new boolean[]{false});

    public final Spec SWIGLU_FLAT = fixed("tl_swiglu_flat_f32", 256, 3,
            new String[]{"alpha", "limit", "n"}, new boolean[]{true, true, false});

    public final Spec Q8_GATE_UP_SWIGLU = tuned("tl_q8_0_gate_up_swiglu_f32_4096", 1,
            new int[]{128, 64, 256, 256}, S_GEMV, 6,
            new String[]{"clamp", "out_dim"}, new boolean[]{true, false}, 4, 2, 8, 4);

    public final Spec MOE_GATE_UP = tuned("tl_moe_gate_up_iq2_xxs_swiglu_4096", 3,
            new int[]{128, 128, 256, 128}, S_MOE, 10,
            new String[]{"clamp", "n_used", "out_dim"}, new boolean[]{true, false, false}, 4, 8, 8, 2);

    public final Spec MOE_DOWN = tuned("tl_moe_down_q2_k_sum_2048", 2,
            new int[]{128, 128, 256, 128}, S_MOE, 4,
            new String[]{"n_used", "out_dim"}, new boolean[]{false, false}, 4, 8, 8, 2);

    public final Spec Q8_GEMV_GROUPED = tuned("tl_q8_0_gemv_grouped_f32_4096", 1,
            new int[]{128, 64, 256, 256}, S_GEMV, 3,
            new String[]{"n_groups", "out_dim"}, new boolean[]{false, false}, 4, 2, 8, 4);

    public final Spec Q8_GEMV_HC_EXPAND4 = tuned("tl_q8_0_gemv_hc_expand4_f32_8192", 1,
            new int[]{256, 128, 512, 512}, S_GEMV, 6,
            new String[]{"out_dim"}, new boolean[]{false}, 4, 2, 8, 4);

    public final Spec Q8_GEMV_SHARED_DOWN_HC_EXPAND4 = tuned(
            "tl_q8_0_gemv_shared_down_hc_expand4_f32_2048", 1,
            new int[]{128, 64, 256, 256}, S_GEMV, 7,
            new String[]{"out_dim"}, new boolean[]{false}, 4, 2, 8, 4);

    public final Spec INDEXER_HADAMARD = fixed("tl_indexer_hadamard_fp4_f32_128", 128, 1,
            new String[]{"row_stride", "rows"}, new boolean[]{false, false});

    public final Spec INDEXER_SCORES = fixed("tl_indexer_scores_f32_64_128", 128, 4,
            new String[]{"comp_len", "n_comp", "n_tokens", "pos0", "ratio", "scale"},
            new boolean[]{false, false, false, false, false, true});

    public final Spec TOPK_RADIX = fixed("tl_topk_radix_select_f32", 256, 2,
            new String[]{"n_rows", "top_k", "width"}, new boolean[]{false, false, false});

    public final Spec HC_WEIGHTED_SUM4 = fixed("tl_hc_weighted_sum4_f32_4096", 256, 3,
            new String[]{"n_tok", "ws"}, new boolean[]{false, false});

    public final Spec BIN_ROWS = fixed("tl_bin_rows_f32", 256, 3,
            new String[]{"a_stride", "b_cols", "b_rows", "b_stride", "dst_stride", "op", "rows", "width"},
            new boolean[]{false, false, false, false, false, false, false, false});

    // =====================================================================
    // Prefill-path kernels (mirror tl_launch_meta.h entries exactly)
    // =====================================================================

    static final String[] S_TL2 = {"default", "t2", "t4"}; // f16_gemv_batch (group 5)
    static final String[] S_FP = {"default", "n64", "k128"};  // flash_prefill (group 11)

    public final Spec QKV_RMSNORM = fixed("tl_qkv_rms_norm_f32_1024_512", 256, 6,
            new String[]{"eps", "rows"}, new boolean[]{true, false});

    public final Spec HC_MIX_GEMM = tuned("tl_hc_mix_gemm_f16_16384", 6,
            new int[]{128, 128, 128, 128}, S_MIX, 3,
            new String[]{"eps", "n_rows"}, new boolean[]{true, false},
            64, 64, 64, 32); // grid x = (n_rows + div-1)/div per variant

    public final Spec F16_GEMV_BATCH_4096 = tuned("tl_f16_gemv_batch_f32_4096", 5,
            new int[]{32, 64, 128}, S_TL2, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec F16_GEMV_BATCH_1024 = tuned("tl_f16_gemv_batch_f32_1024", 5,
            new int[]{32, 64, 128}, S_TL2, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec F16_GEMV_BATCH_16384 = tuned("tl_f16_gemv_batch_f32_16384", 5,
            new int[]{128, 256, 512}, S_TL2, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec F16_GEMV_BATCH4_4096 = fixed("tl_f16_gemv_batch4_f32_4096", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec F16_GEMV_BATCH4_1024 = fixed("tl_f16_gemv_batch4_f32_1024", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec F16_GEMV_BATCH4_16384 = fixed("tl_f16_gemv_batch4_f32_16384", 128, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});

    public final Spec Q8_GEMV_BATCH_4096 = fixed("tl_q8_0_gemv_batch_f32_4096", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH_1024 = fixed("tl_q8_0_gemv_batch_f32_1024", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH_2048 = fixed("tl_q8_0_gemv_batch_f32_2048", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH_8192 = fixed("tl_q8_0_gemv_batch_f32_8192", 64, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH_12288 = fixed("tl_q8_0_gemv_batch_f32_12288", 64, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH_256 = fixed("tl_q8_0_gemv_batch_f32_256", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH4_4096 = fixed("tl_q8_0_gemv_batch4_f32_4096", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH4_1024 = fixed("tl_q8_0_gemv_batch4_f32_1024", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH4_2048 = fixed("tl_q8_0_gemv_batch4_f32_2048", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH4_8192 = fixed("tl_q8_0_gemv_batch4_f32_8192", 64, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH4_12288 = fixed("tl_q8_0_gemv_batch4_f32_12288", 64, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMV_BATCH4_256 = fixed("tl_q8_0_gemv_batch4_f32_256", 32, 3,
            new String[]{"n_tok", "out_dim"}, new boolean[]{false, false});

    public final Spec Q8_GEMM_4096 = tuned("tl_q8_0_gemm_f16x_4096", 0,
            new int[]{64, 64, 128, 128}, S_GEMMV, 3,
            new String[]{"m_pad", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMM_2048 = tuned("tl_q8_0_gemm_f16x_2048", 0,
            new int[]{64, 64, 128, 128}, S_GEMMV, 3,
            new String[]{"m_pad", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMM_1024 = tuned("tl_q8_0_gemm_f16x_1024", 0,
            new int[]{64, 64, 128, 128}, S_GEMMV, 3,
            new String[]{"m_pad", "out_dim"}, new boolean[]{false, false});
    public final Spec Q8_GEMM_8192 = tuned("tl_q8_0_gemm_f16x_8192", 0,
            new int[]{64, 64, 128, 128}, S_GEMMV, 3,
            new String[]{"m_pad", "out_dim"}, new boolean[]{false, false});
    public final Spec F16_GEMM_4096 = tuned("tl_f16_gemm_f16x_4096", 0,
            new int[]{64, 64, 128, 128}, S_GEMMV, 3,
            new String[]{"m_pad", "out_dim"}, new boolean[]{false, false});
    public final Spec F16_GEMM_1024 = tuned("tl_f16_gemm_f16x_1024", 0,
            new int[]{64, 64, 128, 128}, S_GEMMV, 3,
            new String[]{"m_pad", "out_dim"}, new boolean[]{false, false});

    public final Spec F32_TO_F16_4096 = fixed("tl_f32_to_f16_rows_4096", 256, 2,
            new String[]{"m_pad", "n_rows"}, new boolean[]{false, false});
    public final Spec F32_TO_F16_1024 = fixed("tl_f32_to_f16_rows_1024", 256, 2,
            new String[]{"m_pad", "n_rows"}, new boolean[]{false, false});
    public final Spec F32_TO_F16_2048 = fixed("tl_f32_to_f16_rows_2048", 256, 2,
            new String[]{"m_pad", "n_rows"}, new boolean[]{false, false});
    public final Spec F32_TO_F16_8192 = fixed("tl_f32_to_f16_rows_8192", 256, 2,
            new String[]{"m_pad", "n_rows"}, new boolean[]{false, false});
    public final Spec COPY_ROWS_F32 = fixed("tl_copy_rows_f32", 256, 2,
            new String[]{"n_rows", "width"}, new boolean[]{false, false});

    public final Spec FLASH_PREFILL = tuned("tl_flash_prefill_f16_512", 11,
            new int[]{256, 256, 256}, S_FP, 5,
            new String[]{"n_head", "n_keys", "n_rows", "scale"},
            new boolean[]{false, false, false, true});

    public final Spec SET_ROWS_F32 = fixed("tl_set_rows_f32", 256, 3,
            new String[]{"dst_stride4", "rows", "src_stride4", "width"},
            new boolean[]{false, false, false, false});
    public final Spec CPY_F32_3D = fixed("tl_cpy_f32_3d", 256, 2,
            new String[]{"cols", "d1", "d2", "planes", "rows", "s0", "s1", "s2", "vec"},
            new boolean[]{false, false, false, false, false, false, false, false, false});
    public final Spec CPY_TILE_F16_F32 = fixed("tl_cpy_tile_rows_f16_f32", 256, 2,
            new String[]{"n_total", "pos0", "ratio", "width"},
            new boolean[]{false, false, false, false});
    public final Spec CPY_TILE_F32_F32 = fixed("tl_cpy_tile_rows_f32_f32", 256, 2,
            new String[]{"n_total", "pos0", "ratio", "width"},
            new boolean[]{false, false, false, false});
    public final Spec CPY_F16_F32 = fixed("tl_cpy_f16_f32", 256, 2,
            new String[]{"n"}, new boolean[]{false});
    public final Spec CPY_U16 = fixed("tl_cpy_u16", 256, 2,
            new String[]{"n"}, new boolean[]{false});

    public final Spec SORT_I32_ASC_512 = fixed("tl_sort_i32_rows_asc_512", 512, 2,
            new String[]{"rows"}, new boolean[]{false});
    public final Spec ARGSORT_DESC_TOPK_256 = fixed("tl_argsort_desc_topk_f32_256", 256, 2,
            new String[]{"k", "rows"}, new boolean[]{false, false});
    public final Spec GATHER_SCALARS = fixed("tl_gather_scalars_f32", 32, 3,
            new String[]{"dst_stride", "idx_stride", "rows", "table_stride", "width"},
            new boolean[]{false, false, false, false, false});
    public final Spec SUM_ROWS = fixed("tl_sum_rows_f32", 32, 2,
            new String[]{"rows", "src_stride", "width"},
            new boolean[]{false, false, false});
    public final Spec HC_EXPAND4 = fixed("tl_hc_expand4_f32_4096", 256, 6,
            new String[]{"cs1", "cs2", "has_add", "n_tok", "ps"},
            new boolean[]{false, false, false, false, false});

    public final Spec Q8_GEMV_GROUPED_BATCH = fixed("tl_q8_0_gemv_grouped_batch_f32_4096", 32, 3,
            new String[]{"n_groups", "n_tok", "out_dim"}, new boolean[]{false, false, false});
    public final Spec Q8_GEMM_GROUPED = tuned("tl_q8_0_gemm_grouped_f16x_4096", 0,
            new int[]{64, 64, 128, 128}, S_GEMMV, 3,
            new String[]{"c_width_", "m_pad", "n_groups", "out_dim"},
            new boolean[]{false, false, false, false});

    public final Spec MOE_GATE_UP_BATCH = fixed("tl_moe_gate_up_iq2_xxs_swiglu_batch_4096", 128, 10,
            new String[]{"clamp", "n_tok", "n_used", "out_dim"},
            new boolean[]{true, false, false, false});
    public final Spec MOE_DOWN_BATCH = fixed("tl_moe_down_q2_k_sum_batch_2048", 128, 4,
            new String[]{"n_tok", "n_used", "out_dim"}, new boolean[]{false, false, false});
    public final Spec MOE_MAP0 = fixed("tl_moe_map0_6", 256, 4,
            new String[]{"n_expert", "n_tok"}, new boolean[]{false, false});
    public final Spec MOE_PACKED_OFFSETS = fixed("tl_moe_packed_offsets", 256, 2,
            new String[]{"n_expert"}, new boolean[]{false});
    public final Spec MOE_PACK_RHS_F32 = fixed("tl_moe_pack_rhs_f32", 128, 6,
            new String[]{"k_dim", "n_tok_", "n_used", "src_slot_stride4", "src_tok_stride4", "work_cap"},
            new boolean[]{false, false, false, false, false, false});
    public final Spec MOE_PACK_RHS_F16 = fixed("tl_moe_pack_rhs_f16", 128, 6,
            new String[]{"k_dim", "n_tok_", "n_used", "src_slot_stride4", "src_tok_stride4", "work_cap"},
            new boolean[]{false, false, false, false, false, false});
    public final Spec MOE_PACKED_GEMM_IQ2 = fixed("tl_moe_packed_gemm_iq2_xxs_4096", 128, 9,
            new String[]{"n_tok_", "ne0", "rhs_rows_", "work_cap"},
            new boolean[]{false, false, false, false});
    public final Spec MOE_PACKED_GEMM_Q2K = fixed("tl_moe_packed_gemm_q2_k_2048", 128, 7,
            new String[]{"n_tok_", "ne0", "rhs_rows_", "work_cap"},
            new boolean[]{false, false, false, false});
    public final Spec MOE_SWIGLU_WEIGHT_F16 = fixed("tl_moe_swiglu_weight_f16_2048", 256, 4,
            new String[]{"limit", "rows"}, new boolean[]{true, false});
    public final Spec MOE_SUM6 = fixed("tl_moe_sum6_f32_4096", 256, 2,
            new String[]{"tokens"}, new boolean[]{false});
}
