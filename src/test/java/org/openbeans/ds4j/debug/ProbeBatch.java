package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;

/** Prefill raw ids via the batch path and dump pf* tensors for full-precision
 *  comparison. Usage: ProbeBatch <ids-file> <out-dir> */
public class ProbeBatch {
    public static void main(String[] a) throws Exception {
        String src = MetalSources.source();
        Metal metal = Metal.open(src,
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        List<Integer> ids = new ArrayList<>();
        for (String p : Files.readString(Path.of(a[0])).trim().split("\\s+"))
            ids.add(Integer.parseInt(p));
        int maxL = Integer.getInteger("ds4j.maxL", 1);
        Engine e = new Engine(metal, tl, gguf, Integer.getInteger("ds4j.pfCtx", 8192));
        e.debugMaxLayers = maxL;
        e.prefill(ids, true);
        int am = Sampler.argmax(e.logits, e.nVocab);
        System.out.printf("final logits argmax=%d val=%.4f%n", am, e.logits.getFloat(am * 4L));
        Path out = Path.of(a[1]);
        Files.createDirectories(out);
        if (!Boolean.getBoolean("ds4j.pfStream")) {
            dump(e.pfHeads, out.resolve("mine_kqv_back.bin"), (long) ids.size() * 32768);
            dump(e.pfAttnLow, out.resolve("mine_attn_low.bin"), (long) ids.size() * 8192);
            dump(e.pfAttnOut, out.resolve("mine_attn_out.bin"), (long) ids.size() * 4096);
            dump(e.pfQr, out.resolve("mine_q_lora.bin"), (long) ids.size() * 1024);
            dump(e.pfKv, out.resolve("mine_kvcur.bin"), (long) ids.size() * 512);
            dump(e.pfAttnCur, out.resolve("mine_hc_attn_pre.bin"), (long) ids.size() * 4096);
            dump(e.pfAttnNorm, out.resolve("mine_attn_norm.bin"), (long) ids.size() * 4096);
        }
        System.out.println("dumped to " + out);
    }

    static void dump(Metal.Buffer b, Path p, long floats) throws Exception {
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate((int) (floats * 4))
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (long i = 0; i < floats; i++) bb.putFloat((int) (i * 4), b.getFloat(i * 4));
        Files.write(p, bb.array());
    }
}
