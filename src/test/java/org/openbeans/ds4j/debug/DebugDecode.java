package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;

import java.nio.file.Files;
import java.nio.file.Path;

/** Stage-by-stage numeric debugging of the Java decode path. */
public final class DebugDecode {
    static final String PQ = java.util.regex.Pattern.quote("logits") + "\\s*:\\s*\\[" + "([^\\]]*)\\]";
    static float maxdiff(java.nio.file.Path dir, String file, Metal.Buffer buf, int n) throws Exception {
        java.nio.file.Path p = dir.resolve(file);
        if (!java.nio.file.Files.exists(p)) return -1f;
        byte[] raw = java.nio.file.Files.readAllBytes(p);
        int count = (int) Math.min(n, raw.length / 4);
        float maxDiff = 0;
        for (int i = 0; i < count; i++) {
            int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
            float wantF = Float.intBitsToFloat(bits);
            float got = buf.getFloat(i * 4L);
            if (Math.abs(wantF - got) > maxDiff) maxDiff = Math.abs(wantF - got);
        }
        return maxDiff;
    }

    static void cmpWhere(java.nio.file.Path dir, String file, Metal.Buffer buf, int n) throws Exception {
        java.nio.file.Path p = dir.resolve(file);
        if (!java.nio.file.Files.exists(p)) { System.out.println(file + ": (no C dump)"); return; }
        byte[] raw = java.nio.file.Files.readAllBytes(p);
        int count = (int) Math.min(n, raw.length / 4);
        float md = 0; int ma = -1;
        for (int i = 0; i < count; i++) {
            int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
            float wantF = Float.intBitsToFloat(bits);
            float got = buf.getFloat(i * 4L);
            if (Math.abs(wantF - got) > md) { md = Math.abs(wantF - got); ma = i; }
        }
        System.out.printf("  %-42s max diff %f at flat %d (tok %d head %d dim %d): C=%.4f java=%.4f%n",
                file, md, ma, ma / 32768, (ma % 32768) / 512, ma % 512,
                Float.intBitsToFloat((raw[ma*4]&0xff)|(raw[ma*4+1]&0xff)<<8|(raw[ma*4+2]&0xff)<<16|(raw[ma*4+3]&0xff)<<24),
                buf.getFloat(ma * 4L));
    }

    static Metal.Buffer pfBuf(Engine e, String name) { return pfBuf(e, name, 0); }

    static Metal.Buffer pfBuf(Engine e, String name, int il) {
        return switch (name) {
            case "idxCompCache" -> e.idxCompCache[il];
            case "pfCompKv" -> e.pfCompKv;
            case "pfCompSc" -> e.pfCompSc;
            case "pfIndexerQ" -> e.pfIndexerQ;
            case "pfIndexerW" -> e.pfIndexerW;
            case "pfAttnCur" -> e.pfAttnCur;
            case "pfAttnNorm" -> e.pfAttnNorm;
            case "pfQr" -> e.pfQr;
            case "pfKvRaw" -> e.pfKvRaw;
            case "pfKv" -> e.pfKv;
            case "pfHeads" -> e.pfHeads;
            case "pfAttnLow" -> e.pfAttnLow;
            case "pfIndexerScores" -> e.pfIndexerScores.buf;
            case "pfTopk" -> e.pfTopk.buf;
            case "attnCompStage" -> e.attnCompStage;
            case "pfQ" -> e.pfQ;
            case "pfQrNorm" -> e.pfQrNorm;
            case "pfQraw" -> e.dbgQraw;
            case "pfAttnOut" -> e.pfAttnOut;
            case "pfAfterAttnHc" -> e.pfAfterAttnHc;
            case "pfFfnNorm" -> e.pfFfnNorm;
            case "pfRouterProbs" -> e.pfRouterProbs;
            case "pfRouterW" -> e.pfRouterW;
            case "pfRouterSel" -> e.pfRouterSel;
            case "pfRoutedOut" -> e.pfRoutedOut;
            case "pfSharedOut" -> e.pfSharedOut;
            case "pfCurHc" -> e.pfCurHc;
            default -> throw new IllegalArgumentException(name);
        };
    }

    static void cmpRowDist(java.nio.file.Path dir, String file, Metal.Buffer buf, int rows, int w) throws Exception {
        java.nio.file.Path p = dir.resolve(file);
        if (!java.nio.file.Files.exists(p)) { System.out.println("  " + file + ": (no dump)"); return; }
        byte[] raw = java.nio.file.Files.readAllBytes(p);
        int n = (int) Math.min((long) rows * w, raw.length / 4);
        int rw = (int) Math.min(w, raw.length / 4 / Math.max(1, rows));
        java.util.TreeSet<Integer> bad = new java.util.TreeSet<>();
        float md = 0;
        for (int i = 0; i < n; i++) {
            int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
            float want = Float.intBitsToFloat(bits);
            float got = buf.getFloat((long) i * 4);
            float d = Math.abs(want - got);
            if (d > md) md = d;
            if (d > Double.parseDouble(System.getProperty("ds4j.rowth", "0.01"))) bad.add(i / rw);
        }
        System.out.printf("  rowdist %s: max %f, %d/%d rows > 0.01: %s%n", file, md, bad.size(), rows,
                bad.size() <= 40 ? bad.toString() : new java.util.ArrayList<>(bad).subList(0, 40) + " ...");
    }

    static void cmp(java.nio.file.Path dir, String file, Metal.Buffer buf, int n) throws Exception {
        java.nio.file.Path p = dir.resolve(file);
        if (!java.nio.file.Files.exists(p)) { System.out.println("  " + file + ": (no C dump)"); return; }
        byte[] raw = java.nio.file.Files.readAllBytes(p);
        float maxDiff = 0; int at = -1; int nBad = 0; int firstBad = -1;
        int count = Math.min(n, raw.length / 4);
        for (int i = 0; i < count; i++) {
            int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
            float want = Float.intBitsToFloat(bits);
            float got = buf.getFloat((long) i * 4);
            float d = Math.abs(want - got);
            if (d > maxDiff) { maxDiff = d; at = i; }
            if (d > 1e-3) { nBad++; if (firstBad < 0) firstBad = i; }
        }
        System.out.printf("  %-42s max diff %f (%d/%d >1e-3, first at %d)", file, maxDiff, nBad, count, firstBad);
        if (nBad > 0 && System.getProperty("ds4j.cmpDetail") != null) {
            java.util.TreeSet<Integer> badRows = new java.util.TreeSet<>();
            for (int i = 0; i < count; i++) {
                int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
                float want = Float.intBitsToFloat(bits);
                float got = buf.getFloat((long) i * 4);
                if (Math.abs(want - got) > 1e-3) badRows.add(i / Math.max(1, Integer.getInteger("ds4j.cmpRowW", 1)));
            }
            System.out.printf("      bad rows (%d): %s%n", badRows.size(),
                    badRows.size() <= 40 ? badRows.toString() : new java.util.ArrayList<>(badRows).subList(0, 20) + "...");
            int shown = 0;
            for (int i = firstBad; i < count && shown < 12; i++) {
                int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
                float want = Float.intBitsToFloat(bits);
                float got = buf.getFloat((long) i * 4);
                if (Math.abs(want - got) > 1e-3) {
                    System.out.printf("      idx %d: C=%.5f java=%.5f%n", i, want, got);
                    shown++;
                }
            }
        }
        if (maxDiff > 0.01 && at >= 0)
            System.out.printf(" at %d (C=%.5f java=%.5f)", at,
                    Float.intBitsToFloat((raw[at*4]&0xff)|(raw[at*4+1]&0xff)<<8|(raw[at*4+2]&0xff)<<16|(raw[at*4+3]&0xff)<<24),
                    buf.getFloat((long) at * 4));
        System.out.println();
    }

    /** Strided compare: only the first `live` floats of each `stride` block. */
    static void cmpStrided(java.nio.file.Path dir, String file, Metal.Buffer buf, int rows, int stride, int live) throws Exception {
        java.nio.file.Path p = dir.resolve(file);
        if (!java.nio.file.Files.exists(p)) { System.out.println("  " + file + ": (no C dump)"); return; }
        byte[] raw = java.nio.file.Files.readAllBytes(p);
        int nRows = Math.min(rows, raw.length / 4 / stride);
        float maxDiff = 0; int at = -1; int badRows = 0;
        for (int r = 0; r < nRows; r++) {
            boolean rowBad = false;
            for (int j = 0; j < live; j++) {
                int i = r * stride + j;
                int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
                float want = Float.intBitsToFloat(bits);
                float got = buf.getFloat((long) i * 4);
                float d = Math.abs(want - got);
                if (d > maxDiff) { maxDiff = d; at = i; }
                if (d > 1e-6) rowBad = true;
            }
            if (rowBad) badRows++;
        }
        System.out.printf("  %-42s max diff %f (%d/%d rows >1e-6)", file, maxDiff, badRows, nRows);
        if (maxDiff > 0.01 && at >= 0)
            System.out.printf(" at %d (C=%.5f java=%.5f)", at,
                    Float.intBitsToFloat((raw[at*4]&0xff)|(raw[at*4+1]&0xff)<<8|(raw[at*4+2]&0xff)<<16|(raw[at*4+3]&0xff)<<24),
                    buf.getFloat((long) at * 4));
        System.out.println();
    }

    static void cmpF16Row(java.nio.file.Path dir, String file, Metal.Buffer buf, long rowBytes) throws Exception {
        java.nio.file.Path p = dir.resolve(file);
        if (!java.nio.file.Files.exists(p)) { System.out.println("  " + file + ": (no C dump)"); return; }
        byte[] raw = java.nio.file.Files.readAllBytes(p);
        float maxDiff = 0;
        int count = (int) Math.min(512, raw.length / 4);
        for (int i = 0; i < count; i++) {
            int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
            float wantF = Float.intBitsToFloat(bits);
            float got = buf.getF16(rowBytes + i * 2L);
            if (Math.abs(wantF - got) > maxDiff) maxDiff = Math.abs(wantF - got);
        }
        System.out.printf("  %-42s max diff %f%n", file, maxDiff);
        if (maxDiff > 0.05) {
            for (int i = 0; i < count; i++) {
                int bits = (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
                float wantF = Float.intBitsToFloat(bits);
                float got = buf.getF16(rowBytes + i * 2L);
                if (Math.abs(wantF - got) > 0.05)
                    System.out.printf("    [%d] C=%.5f java=%.5f%n", i, wantF, got);
            }
        }
    }

    static void cmpI32(java.nio.file.Path dir, String file, Metal.Buffer buf, int n) throws Exception {
        java.nio.file.Path p = dir.resolve(file);
        if (!java.nio.file.Files.exists(p)) { System.out.println("  " + file + ": (no C dump)"); return; }
        byte[] raw = java.nio.file.Files.readAllBytes(p);
        StringBuilder got = new StringBuilder(), want = new StringBuilder();
        for (int i = 0; i < Math.min(n, raw.length / 4); i++) {
            got.append(e_int(raw, i)).append(' ');
            want.append(buf.getInt(i * 4L)).append(' ');
        }
        System.out.printf("  %-42s C:[%s] java:[%s]%n", file, want, got);
    }

    static int e_int(byte[] raw, int i) {
        return (raw[i*4]&0xff) | (raw[i*4+1]&0xff)<<8 | (raw[i*4+2]&0xff)<<16 | (raw[i*4+3]&0xff)<<24;
    }

    public static void main(String[] args) throws Exception {
        String root = Files.exists(Path.of("..", "tests", "test-vectors")) ? ".." : "../..";
        String model = System.getProperty("ds4j.model", "ds4flash.gguf");
        String src = MetalSources.source();
        Metal metal = Metal.open(src,
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(model);
        Engine e = new Engine(metal, tl, gguf, 1024);

        int token = 0; // BOS, matches the C dump at pos 0 for prompt "Hi"
        System.out.println("=== stage 1: embedding ===");
        try (Metal.Batch bb = metal.begin()) {
            tl.launch(bb, tl.GET_ROWS_F16_F32, 4, 1, 1,
                    tl.args(tl.GET_ROWS_F16_F32).i(1024).i(1).i(1024).i(4096),
                    Tl.B.b(e.embedRow, 0), Tl.B.inline(token), e.mb(e.tokEmbOff));
            bb.commit();
            bb.waitGpu();
        }
        // CPU reference: token_embd row, f16 -> f32
        Gguf.Tensor te = gguf.t("token_embd.weight");
        float[] want = new float[8];
        try (java.nio.channels.FileChannel fc = java.nio.channels.FileChannel.open(
                java.nio.file.Path.of(model))) {
            java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(16);
            fc.read(b, te.absOffset() + (long) token * 4096 * 2);
            b.flip();
            for (int i = 0; i < 8; i++)
                want[i] = Metal.Buffer.halfToFloat(b.getShort() & 0xffff);
        }
        System.out.print("embedRow got :");
        for (int i = 0; i < 8; i++) System.out.printf(" %.5f", e.embedRow.getFloat(i * 4L));
        System.out.print("\nembedRow want:");
        for (int i = 0; i < 8; i++) System.out.printf(" %.5f", want[i]);
        System.out.println();

        System.out.println("=== stage 2: repeat into cur_hc ===");
        try (Metal.Batch bb = metal.begin()) {
            tl.launch(bb, tl.REPEAT_ROWS, 4, 4, 1,
                    tl.args(tl.REPEAT_ROWS).i(1024).i(4096).i(4).i(1).i(1024).i(4096),
                    Tl.B.b(e.curHc, 0), Tl.B.b(e.embedRow, 0));
            bb.commit();
            bb.waitGpu();
        }
        System.out.print("curHc[1][0..4]:");
        for (int i = 0; i < 4; i++) System.out.printf(" %.5f", e.curHc.getFloat(4096 * 4L + i * 4L));
        System.out.println();

        System.out.println("=== full decode token " + token + " ===");
        String dump = System.getProperty("ds4j.dump", "/tmp/dump");
        java.nio.file.Path dir = java.nio.file.Path.of(dump);
        Tokenizer tok = new Tokenizer(gguf, e.nVocab);
        java.util.List<Integer> chat = tok.encodeChat("You are a helpful assistant",
                "Hello world, this is a test of the compressor state machine for several tokens.", false);
        System.out.println("chat tokens: " + chat);
        // walk the prompt; after each token compare layer-0 dumps at that pos
        int cmpPos = Integer.parseInt(System.getProperty("ds4j.pos", "1"));
        int maxLayer = Integer.parseInt(System.getProperty("ds4j.maxLayer", "3"));
        if (System.getProperty("ds4j.pfTokens") != null) {
            java.nio.file.Path dirT = java.nio.file.Path.of(System.getProperty("ds4j.dump", "/tmp/dumptool"));
            int ctxT = Integer.parseInt(System.getProperty("ds4j.pfCtx", "16384"));
            String[] parts = java.nio.file.Files.readString(java.nio.file.Path.of(
                    System.getProperty("ds4j.pfTokens"))).trim().split("[\\s]+");
            java.util.List<Integer> idsT = new java.util.ArrayList<>();
            for (String p2 : parts) idsT.add(Integer.parseInt(p2));
            int nT = idsT.size();
            int pfLayersT = Integer.parseInt(System.getProperty("ds4j.pfLayers", "43"));
            int pfFirst = Integer.parseInt(System.getProperty("ds4j.pfFirst", "1"));
            String pfStartS = System.getProperty("ds4j.pfStart");
            if (pfStartS != null) {
                // resume-boundary parity, CLI-interactive shape: fresh prefill
                // [0, F), decode the generated token at pos F, then extend the
                // checkpoint with an S-token suffix (batch path, pos0 = F+1).
                int F = Integer.parseInt(System.getProperty("ds4j.pfFresh", "0"));
                int S = Integer.parseInt(System.getProperty("ds4j.pfSuffix", "1"));
                int genTok = Integer.parseInt(System.getProperty("ds4j.pfGenTok", "1"));
                int P = F + 1;
                for (int L = pfFirst; L <= pfLayersT; L++) {
                    int il = L - 1;
                    e = new Engine(metal, tl, gguf, ctxT);
                    e.debugMaxLayers = L;
                    e.prefill(idsT.subList(0, F), false);
                    try (Metal.Batch bb = metal.begin()) {
                        e.decode(bb, genTok, F, false);
                    }
                    cmp(dirT, "d_attn_out-" + il + "_pos" + F + ".bin", e.attnOut, 4096);
                    java.util.List<Integer> suf = idsT.subList(P, P + S);
                    if (S >= e.RESUME_BATCH_MIN_TOKENS) {
                        e.prefillFrom(P, suf, false);
                        cmp(dirT, "d_attn_out-" + il + "_pos" + P + ".bin", e.pfAttnOut, S * 4096);
                    } else {
                        for (int k = 0; k < S; k++) {
                            try (Metal.Batch bb = metal.begin()) {
                                e.decode(bb, suf.get(k), P + k, false);
                            }
                            cmp(dirT, "d_attn_out-" + il + "_pos" + (P + k) + ".bin", e.attnOut, 4096);
                        }
                    }
                }
                return;
            }
            String pfGenS = System.getProperty("ds4j.pfGen");
            if (pfGenS != null) {
                // decode-after-prefill parity: greedy steps, compare attn_out
                // per layer at each decode position (fresh engine per layer)
                int gen = Integer.parseInt(pfGenS);
                String genTokFile = System.getProperty("ds4j.pfGenTokens");
                int[] genToks = null;
                if (genTokFile != null) {
                    String[] gp = java.nio.file.Files.readString(java.nio.file.Path.of(genTokFile)).trim().split("[\\s]+");
                    genToks = new int[gp.length];
                    for (int i = 0; i < gp.length; i++) genToks[i] = Integer.parseInt(gp[i]);
                    if (gen > genToks.length) gen = genToks.length;
                }
                for (int L = pfFirst; L <= pfLayersT; L++) {
                    int il = L - 1;
                    e = new Engine(metal, tl, gguf, ctxT);
                    e.debugMaxLayers = L;
                    e.prefill(idsT, true);
                    if (System.getProperty("ds4j.pfLogits") != null) {
                        java.nio.file.Path lp = dirT.resolve("d_logits-" + (nT - 1) + ".bin");
                        java.nio.ByteBuffer lb = java.nio.ByteBuffer.allocate(e.nVocab * 4)
                                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                        for (int i = 0; i < e.nVocab; i++) lb.putFloat(i * 4, e.logits.getFloat(i * 4L));
                        java.nio.file.Files.write(lp, lb.array());
                        int am = Sampler.argmax(e.logits, e.nVocab);
                        System.out.printf("  prefill logits argmax=%d val=%.4f lg0=%.4f lg1=%.4f%n",
                                am, e.logits.getFloat(am * 4L),
                                e.logits.getFloat(0), e.logits.getFloat(4));
                    }
                    int tokId = Sampler.argmax(e.logits, e.nVocab);
                    for (int step = 0; step < gen; step++) {
                        int pos = nT + step;
                        String tag = System.getProperty("ds4j.pfTag", String.valueOf(il));
                        int stepTok = genToks != null ? genToks[step] : tokId;
                        try (Metal.Batch bb = metal.begin()) {
                            e.decode(bb, stepTok, pos, true);
                        }
                        if (System.getProperty("ds4j.pfLogits") != null) {
                            java.nio.file.Path lp = dirT.resolve("d_logits-" + pos + ".bin");
                            java.nio.ByteBuffer lb = java.nio.ByteBuffer.allocate(e.nVocab * 4)
                                    .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                            for (int i = 0; i < e.nVocab; i++) lb.putFloat(i * 4, e.logits.getFloat(i * 4L));
                            java.nio.file.Files.write(lp, lb.array());
                        }
                        cmp(dirT, "d_attn_out-" + tag + "_pos" + pos + ".bin", e.attnOut, 4096);
                        if (System.getProperty("ds4j.pfStages") != null) {
                            cmp(dirT, "d_hc_attn_pre-" + tag + "_pos" + pos + ".bin", e.attnCur, 4096);
                            cmp(dirT, "d_attn_norm-" + tag + "_pos" + pos + ".bin", e.attnNorm, 4096);
                            cmp(dirT, "d_q_lora-" + tag + "_pos" + pos + ".bin", e.qr, 1024);
                            cmp(dirT, "d_KVraw-" + tag + "_pos" + pos + ".bin", e.kvRaw, 512);
                            cmp(dirT, "d_KVcur-" + tag + "_pos" + pos + ".bin", e.kv, 512);
                            cmpStrided(dirT, "d_kqv_back-" + tag + "_pos" + pos + ".bin", e.heads, 1, 32768, 1024);
                            cmp(dirT, "d_attn_low-" + tag + "_pos" + pos + ".bin", e.attnLow, 8192);
                            cmp(dirT, "d_hc_ffn_post-" + tag + "_pos" + pos + ".bin", e.curHc, 16384);
                            // result_* dumps carry pos=0 and are overwritten
                            // per decode step: valid for the final step only
                            cmp(dirT, "d_result_hc-43_pos0.bin", e.outputEmb, 4096);
                            cmp(dirT, "d_result_hc_weights-43_pos0.bin", e.outputPre, 4);
                            cmp(dirT, "d_result_norm-43_pos0.bin", e.outputNorm, 4096);
                            cmp(dirT, "d_result_output-43_pos0.bin", e.logits, e.nVocab);
                        }
                        tokId = genToks != null ? genToks[step] : Sampler.argmax(e.logits, e.nVocab);
                    }
                    if (System.getProperty("ds4j.pfProbe") != null && genToks != null) {
                        Engine f = new Engine(metal, tl, gguf, ctxT);
                        f.debugMaxLayers = 1;
                        try (Metal.Batch bb = metal.begin()) {
                            f.decode(bb, genToks[0], 0, false);
                        }
                        System.out.print("  probe fresh  attnCur:");
                        for (int i = 0; i < 6; i++) System.out.printf(" %.5f", f.attnCur.getFloat(i * 4L));
                        System.out.print(" | fresh embedRow:");
                        for (int i = 0; i < 6; i++) System.out.printf(" %.5f", f.embedRow.getFloat(i * 4L));
                        System.out.println();
                        System.out.print("  probe after attnCur:");
                        for (int i = 0; i < 6; i++) System.out.printf(" %.5f", e.attnCur.getFloat(i * 4L));
                        System.out.print(" | after embedRow:");
                        for (int i = 0; i < 6; i++) System.out.printf(" %.5f", e.embedRow.getFloat(i * 4L));
                        System.out.print(" | after hcMix:");
                        for (int i = 0; i < 6; i++) System.out.printf(" %.5f", e.hcMix.getFloat(i * 4L));
                        System.out.println();
                    }
                }
                return;
            }
            for (int L = pfFirst; L <= pfLayersT; L++) {
                int il = L - 1;
                String[][] compares = {
                        {"d_hc_attn_pre", "pfAttnCur", "4096"},
                        {"d_attn_norm", "pfAttnNorm", "4096"},
                        {"d_q_lora", "pfQr", "1024"},
                        {"d_KVraw", "pfKvRaw", "512"},
                        {"d_KVcur", "pfKv", "512"},
                        {"d_kqv_out", "pfHeads", "32768"},
                        {"d_kqv_back", "pfHeads", "32768"},
                        {"d_KVcompress", "attnCompStage", "512"},
                        {"d_indexer_KVcompress", "idxCompCache", "128"},
                        {"d_indexer_q", "pfIndexerQ", "8192"},
                        {"d_indexer_w", "pfIndexerW", "64"},
                        {"d_indexer_comp_kv_raw", "pfCompKv", "256"},
                        {"d_indexer_comp_score_raw", "pfCompSc", "256"},
                        {"d_indexer_scores", "pfIndexerScores", "16000000"},
                        {"d_indexer_topk", "pfTopk", "16000000"},
                        {"d_attn_low", "pfAttnLow", "8192"},
                        {"d_attn_out", "pfAttnOut", "4096"},
                        {"d_KVcur", "pfKv", "512"},
                        {"d_hc_attn_post", "pfAfterAttnHc", "16384"},
                        {"d_ffn_norm", "pfFfnNorm", "4096"},
                        {"d_ffn_moe_probs", "pfRouterProbs", "256"},
                        {"d_ffn_moe_weights_scaled", "pfRouterW", "6"},
                        {"d_ffn_moe_topk", "pfRouterSel", "6"},
                        {"d_ffn_moe_out", "pfRoutedOut", "4096"},
                        {"d_ffn_shexp", "pfSharedOut", "4096"},
                        {"d_hc_ffn_post", "pfCurHc", "16384"},
                };
                // pass A: chunk 1 only (pos0 dumps); pass B: full prompt
                // (pos4096 dumps) — fresh engine each pass
                for (String pass : new String[]{"A", "B"}) {
                    String posTag = pass.equals("A") ? "_pos0"
                            : "_pos" + (((nT - 1) / 4096) * 4096);
                    java.util.List<Integer> slice = pass.equals("A")
                            ? idsT.subList(0, Math.min(4096, nT)) : idsT;
                    int nC2 = slice.size() == nT && nT > 4096 ? nT - 4096 : slice.size();
                    if (pass.equals("B") && nT <= 4096) break;
                    e = new Engine(metal, tl, gguf, ctxT);
                    e.debugMaxLayers = L;
                    e.prefill(slice, false);
                    for (String[] c : compares) {
                        if (c[0].startsWith("d_kqv_"))
                            cmpStrided(dirT, c[0] + "-" + il + posTag + ".bin", pfBuf(e, c[1], il),
                                    nC2, 32768, 1024);
                        else
                            cmp(dirT, c[0] + "-" + il + posTag + ".bin", pfBuf(e, c[1], il),
                                    nC2 * Integer.parseInt(c[2]));
                    }
                    String rd = System.getProperty("ds4j.rowdist");
                    if (rd != null && pass.equals("B")) {
                        String file = switch (rd) {
                            case "pfAttnCur" -> "d_hc_attn_pre"; case "pfAttnNorm" -> "d_attn_norm";
                            case "pfQr" -> "d_q_lora"; case "pfKvRaw" -> "d_KVraw"; case "pfKv" -> "d_KVcur";
                            case "pfHeads" -> "d_kqv_back"; case "pfQ" -> "d_Qcur"; case "pfAttnOut" -> "d_attn_out";
                            case "pfQrNorm" -> "d_q_lora_norm"; case "pfQraw" -> "d_Qraw";
                            case "pfAfterAttnHc" -> "d_hc_attn_post"; case "pfFfnNorm" -> "d_ffn_norm";
                            case "pfRoutedOut" -> "d_ffn_moe_out"; case "pfCurHc" -> "d_hc_ffn_post";
                            default -> throw new IllegalArgumentException(rd);
                        };
                        int w2 = switch (rd) { case "pfQr" -> 1024; case "pfKvRaw", "pfKv" -> 512; default -> 4096; };
                        if (rd.equals("pfHeads")) w2 = 32768;
                        if (rd.equals("pfKv")) w2 = 512;
                        if (rd.equals("pfQ")) w2 = 32768;
                        if (rd.equals("pfAfterAttnHc") || rd.equals("pfCurHc")) w2 = 16384;
                        cmpRowDist(dirT, file + "-" + il + posTag + ".bin", pfBuf(e, rd), nC2, w2);
                    }
                }
            }
            return;
        }
        if (System.getProperty("ds4j.pfdump") != null) {
            // chunked prefill vs the C engine's layer-major batch dumps
            java.nio.file.Path dir0 = java.nio.file.Path.of(System.getProperty("ds4j.dump", "/tmp/dump"));
            String pfPromptFile = System.getProperty("ds4j.pfPrompt");
            String pfPrompt = pfPromptFile != null
                    ? java.nio.file.Files.readString(java.nio.file.Path.of(pfPromptFile))
                    : "Say hello and tell me what model you are.";
            java.util.List<Integer> chatPf = tok.encodeChat(
                    System.getProperty("ds4j.pfSystem", "You are a helpful assistant"),
                    pfPrompt, false);
            System.out.println("pf tokens: " + chatPf.size());
            int n = chatPf.size();
            int pfLayers = Integer.parseInt(System.getProperty("ds4j.pfLayers", "1"));
            int pfStep = Integer.parseInt(System.getProperty("ds4j.pfStep", "1"));
            for (int L = 1; L <= pfLayers; L += pfStep) {
                e.debugMaxLayers = L;
                e.prefill(chatPf, false);
                int il = L - 1;
                if (System.getProperty("ds4j.moeXcheck") != null && n >= 32 && il == 0) {
                    e.moeXcheck(0, n);
                    e.moeXcheckStage(0, n);
                }
                        cmp(dir0, "d_hc_attn_pre-" + il + "_pos0.bin", e.pfAttnCur, n * 4096);
                        cmp(dir0, "d_attn_norm-" + il + "_pos0.bin", e.pfAttnNorm, n * 4096);
                        cmp(dir0, "d_q_lora-" + il + "_pos0.bin", e.pfQr, n * 1024);
                        cmp(dir0, "d_KVraw-" + il + "_pos0.bin", e.pfKvRaw, n * 512);
                        cmp(dir0, "d_KVnorm-" + il + "_pos0.bin", e.pfKv, n * 512);
                        cmp(dir0, "d_KVrope-" + il + "_pos0.bin", e.pfKv, n * 512);
                        cmp(dir0, "d_KVcur-" + il + "_pos0.bin", e.pfKv, n * 512);
                        cmp(dir0, "d_Qraw-" + il + "_pos0.bin", e.pfQ, n * 32768);
                        cmp(dir0, "d_Qcur-" + il + "_pos0.bin", e.pfQ, n * 32768);
                        cmpWhere(dir0, "d_kqv_out-" + il + "_pos0.bin", e.pfHeads, n * 32768);
                        if (il >= 2 && il % 2 == 0) {
                    cmp(dir0, "d_attn_comp_kv_raw-" + il + "_pos0.bin", e.pfCompKv, n * 1024);
                    cmp(dir0, "d_attn_comp_score_raw-" + il + "_pos0.bin", e.pfCompSc, n * 1024);
                    int nCompRow = n / 4;
                    cmp(dir0, "d_KVcompress-" + il + "_pos0.bin", e.attnCompStage, nCompRow * 512);
                    cmp(dir0, "d_attn_state_kv-" + il + "_pos0.bin", e.attnStateKv[il], 8 * 1024);
                    cmp(dir0, "d_indexer_KVcompress-" + il + "_pos0.bin", e.idxCompCache[il], nCompRow * 128);
                }
                cmp(dir0, "d_kqv_out-" + il + "_pos0.bin", e.pfHeads, n * 32768);
                        cmp(dir0, "d_kqv_back-" + il + "_pos0.bin", e.pfHeads, n * 32768);
                        cmp(dir0, "d_attn_low-" + il + "_pos0.bin", e.pfAttnLow, n * 8192);
                        cmp(dir0, "d_attn_out-" + il + "_pos0.bin", e.pfAttnOut, n * 4096);
                        cmp(dir0, "d_hc_attn_post-" + il + "_pos0.bin", e.pfAfterAttnHc, n * 16384);
                        cmp(dir0, "d_ffn_norm-" + il + "_pos0.bin", e.pfFfnNorm, n * 4096);
                        cmp(dir0, "d_ffn_moe_logits-" + il + "_pos0.bin", e.pfRouterLogits, n * 256);
                        cmp(dir0, "d_ffn_moe_probs-" + il + "_pos0.bin", e.pfRouterProbs, n * 256);
                        cmpI32(dir0, "d_ffn_moe_topk-" + il + "_pos0.i32", e.pfRouterSel, n * 6);
                        cmp(dir0, "d_ffn_moe_weights_scaled-" + il + "_pos0.bin", e.pfRouterW, n * 6);
                        cmp(dir0, "d_ffn_moe_out-" + il + "_pos0.bin", e.pfRoutedOut, n * 4096);
                        cmp(dir0, "d_ffn_shexp-" + il + "_pos0.bin", e.pfSharedOut, n * 4096);
                cmp(dir0, "d_hc_ffn_post-" + il + "_pos0.bin", e.pfCurHc, n * 16384);
            
            }
return;
        }
        if (System.getProperty("ds4j.official") != null) {
            // verify greedy tokens against the official DeepSeek API vectors
            java.util.List<String[]> cases = new java.util.ArrayList<>();
            cases.add(new String[]{"short_reasoning_plain", "Answer with only the number: 2048 divided by 128 is", "16"});
            cases.add(new String[]{"short_italian_fact", officialPrompt(root, "short_italian_fact.txt"), null});
            cases.add(new String[]{"short_code_completion", officialPrompt(root, "short_code_completion.txt"), null});
            for (String[] c : cases) {
                e = new Engine(metal, tl, gguf, 1024); // fresh state per case, like ds4_test's session
                java.util.List<Integer> ids = tok.encodeChat("", c[1], false);
                StringBuilder out = new StringBuilder();
                int tokId = 0;
                // decode stream (bit-verified path); prefill parity is
                // tracked separately via the pfdump layer harness
                for (int pos = 0; pos < ids.size(); pos++) {
                    try (Metal.Batch bb = metal.begin()) {
                        e.decode(bb, ids.get(pos), pos, pos == ids.size() - 1);
                    }
                }
                for (int step = 0; step < 4; step++) {
                    Metal.Buffer lg = e.logits;
                    int next = Sampler.argmax(lg, e.nVocab);
                    if (step == 0) {
                        int[] top = new int[5]; float[] tv = new float[5];
                        for (int i = 0; i < e.nVocab; i++) {
                            float v = lg.getFloat(i * 4L);
                            for (int j = 0; j < 5; j++) if (tv[j] < v) {
                                for (int k = 4; k > j; k--) { tv[k] = tv[k-1]; top[k] = top[k-1]; }
                                tv[j] = v; top[j] = i; break;
                            }
                        }
                        for (int j = 0; j < 5; j++)
                            System.out.printf("  top%d id=%d %.4f [%s]%n", j, top[j], tv[j], tok.decodeToken(top[j]));
                    }
                    if (next == tok.eosId) break;
                    out.append(tok.decodeToken(next));
                    tokId = next;
                    try (Metal.Batch bb = metal.begin()) {
                        e.decode(bb, tokId, ids.size() + step, true);
                    }
                }
                if (ids.size() >= 19)
                    System.out.println("  (prefill parity: see ds4j.pfdump; this check decodes)");
                System.out.println("[" + c[0] + "] ids " + ids.size() + " first-tok " + (ids.isEmpty() ? -1 : ids.get(0)) + " greedy: " + out);
            }
            return;
        }
        if (System.getProperty("ds4j.gen") != null) {
            java.util.List<Integer> chat2 = tok.encodeChat("You are a helpful assistant",
                    "Say hello and tell me what model you are.", false);
            int tokId = 0;
            StringBuilder txt = new StringBuilder();
            for (int pos = 0; pos < chat2.size() + 20; pos++) {
                e.debugMaxLayers = -1;
                Metal.Buffer lg;
                try (Metal.Batch bb = metal.begin()) {
                    lg = e.decode(bb, pos < chat2.size() ? chat2.get(pos) : tokId, pos, true);
                }
                int next = Sampler.argmax(lg, e.nVocab);
                if (pos < chat2.size() - 1) continue;
                if (pos == chat2.size() - 1) {
                    // compare the full logits vector with the C engine dump
                    String cj = java.nio.file.Files.readString(java.nio.file.Path.of("/tmp/c_logits.json"));
                    int ls = cj.indexOf("\"logits\"");
                    if (ls >= 0) {
                        int lb = cj.indexOf('[', ls);
                        if (lb < 0) { ls = -1; } else {
                        int le = cj.indexOf(']', lb);
                        String[] parts = cj.substring(lb + 1, le).replace("\n", "").split(",");
                        float md = 0; int ma = -1;
                        for (int i = 0; i < Math.min(parts.length, e.nVocab); i++) {
                            float w2 = Float.parseFloat(parts[i].trim());
                            float g2 = lg.getFloat(i * 4L);
                            if (Math.abs(w2 - g2) > md) { md = Math.abs(w2 - g2); ma = i; }
                        }
                        System.out.printf("logits vs C: max diff %.4f at id %d (argmax java=%d)%n", md, ma, next);
                        }
                    }
                }
                tokId = next;
                txt.append(tok.decodeToken(next)).append('|');
            }
            System.out.println("greedy: " + txt);
            return;
        }
        // decode the prompt up to cmpPos, then per layer-count re-decode so the
        // layer under comparison is always the last one that ran (scratch
        // buffers are shared across layers)
        for (int pos = 0; pos <= cmpPos; pos++) {
            for (int L = 1; L <= maxLayer; L++) {
                e.debugMaxLayers = L;
                try (Metal.Batch bb = metal.begin()) {
                    e.decode(bb, chat.get(pos), pos, false);
                }
                int il = L - 1;
                System.out.printf("pos %d layer %d:%n", pos, il);
                cmp(dir, "d_hc_attn_pre-" + il + "_pos" + pos + ".bin", e.attnCur, 4096);
                cmp(dir, "d_attn_norm-" + il + "_pos" + pos + ".bin", e.attnNorm, 4096);
                cmp(dir, "d_q_lora-" + il + "_pos" + pos + ".bin", e.qr, 1024);
                cmp(dir, "d_KVraw-" + il + "_pos" + pos + ".bin", e.kvRaw, 512);
                cmp(dir, "d_Qcur-" + il + "_pos" + pos + ".bin", e.q, 32768);
                cmp(dir, "d_kqv_back-" + il + "_pos" + pos + ".bin", e.heads, 32768);
                cmp(dir, "d_attn_out-" + il + "_pos" + pos + ".bin", e.attnOut, 4096);
                cmp(dir, "d_hc_attn_post-" + il + "_pos" + pos + ".bin", e.afterAttnHc, 16384);
                cmp(dir, "d_ffn_norm-" + il + "_pos" + pos + ".bin", e.ffnNorm, 4096);
                cmp(dir, "d_ffn_moe_probs-" + il + "_pos" + pos + ".bin", e.routerProbs, 256);
                cmpI32(dir, "d_ffn_moe_topk-" + il + "_pos" + pos + ".i32", e.routerSelected, 6);
                cmp(dir, "d_ffn_moe_weights_scaled-" + il + "_pos" + pos + ".bin", e.routerWeights, 6);
                cmp(dir, "d_ffn_moe_out-" + il + "_pos" + pos + ".bin", e.routedOut, 4096);
                cmp(dir, "d_ffn_shexp-" + il + "_pos" + pos + ".bin", e.sharedOut, 4096);
                cmp(dir, "d_hc_ffn_post-" + il + "_pos" + pos + ".bin", e.curHc, 16384);
                if (pos == 0 && il == 28) {
                    cmp(dir, "d_hc_attn_pre_mixes-28_pos0.bin", e.hcMix, 24);
                }
                if (pos == 7 && il == 2) {
                    cmpF16Row(dir, "d_KVcompress-2_pos7.bin", e.attnCompCache[il], (long) 1 * 512 * 2);
                    cmpF16Row(dir, "d_KVcompress-2_pos7.bin", e.attnCompCache[il], 0);
                    for (int t = 4; t <= 7; t++) {
                        byte[] rk = java.nio.file.Files.readAllBytes(dir.resolve("d_attn_comp_kv_raw-2_pos" + t + ".bin"));
                        int d = 512;
                        int bits = (rk[d*4]&0xff) | (rk[d*4+1]&0xff)<<8 | (rk[d*4+2]&0xff)<<16 | (rk[d*4+3]&0xff)<<24;
                        System.out.printf("  post-shift: state r%d[512] C=%.5f java=%.5f%n",
                                t - 4, Float.intBitsToFloat(bits),
                                e.attnStateKv[il].getFloat(((long)(t-4)*1024 + 512) * 4));
                    }
                }
            }
        }
        if (true) return;
        int maxL = Integer.parseInt(System.getProperty("ds4j.maxL", "4"));
        for (int L = 1; L <= maxL; L++) {
            e.debugMaxLayers = L;
            try (Metal.Batch bb = metal.begin()) {
                e.decode(bb, token, 0, true);
            }
            int il = L - 1;
            System.out.printf("== layer %d ==%n", il);
            cmp(dir, "d_attn_norm-" + il + "_pos0.bin", e.attnNorm, 4096);
            cmp(dir, "d_q_lora-" + il + "_pos0.bin", e.qr, 1024);
            cmp(dir, "d_kqv_out-" + il + "_pos0.bin", e.heads, 32768);
            cmp(dir, "d_attn_low-" + il + "_pos0.bin", e.attnLow, 8192);
            cmp(dir, "d_attn_out-" + il + "_pos0.bin", e.attnOut, 4096);
            cmp(dir, "d_hc_attn_post-" + il + "_pos0.bin", e.afterAttnHc, 16384);
            cmp(dir, "d_ffn_norm-" + il + "_pos0.bin", e.ffnNorm, 4096);
            cmp(dir, "d_ffn_moe_logits-" + il + "_pos0.bin", e.routerLogits, 256);
            cmp(dir, "d_ffn_moe_probs-" + il + "_pos0.bin", e.routerProbs, 256);
            cmpI32(dir, "d_ffn_moe_topk-" + il + "_pos0.i32", e.routerSelected, 6);
            cmp(dir, "d_ffn_moe_weights_scaled-" + il + "_pos0.bin", e.routerWeights, 6);
            cmp(dir, "d_ffn_moe_out-" + il + "_pos0.bin", e.routedOut, 4096);
            cmp(dir, "d_ffn_shexp-" + il + "_pos0.bin", e.sharedOut, 4096);
            cmp(dir, "d_hc_ffn_post-" + il + "_pos0.bin", e.curHc, 16384); // swapped after layer
            if (il == 3) {
                java.nio.file.Path pp = dir.resolve("d_ffn_moe_probs-3_pos0.bin");
                byte[] raw = java.nio.file.Files.readAllBytes(pp);
                for (int id : new int[]{79, 124, 143, 153, 175, 213, 230}) {
                    int bits = (raw[id*4]&0xff) | (raw[id*4+1]&0xff)<<8 | (raw[id*4+2]&0xff)<<16 | (raw[id*4+3]&0xff)<<24;
                    System.out.printf("  probs[%d] C=%.7f java=%.7f%n", id, Float.intBitsToFloat(bits), e.routerProbs.getFloat(id*4L));
                }
            }
        }
    }

    /** Official-vector prompt fixture; lives in the upstream ds4 checkout's
     *  tests/test-vectors tree, which -Dds4j.official runs depend on. */
    static String officialPrompt(String root, String file) {
        java.nio.file.Path p = java.nio.file.Path.of(root, "tests/test-vectors/flash-0731/prompts", file);
        if (!java.nio.file.Files.exists(p))
            throw new IllegalStateException("missing official-vector fixture " + p
                    + " — the tests/test-vectors/flash-0731 prompts live in the"
                    + " upstream ds4 checkout this harness was built against");
        try {
            return new String(java.nio.file.Files.readAllBytes(p), java.nio.charset.StandardCharsets.UTF_8);
        } catch (java.io.IOException e) { throw new RuntimeException(e); }
    }

}
