package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;
/** Split-decode per-token breakdown: wall vs GPU-busy (Metal timestamps)
 *  vs the C's known ~22.0ms/token. */
public class SplitTiming {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 0);
        List<Integer> ids = tok.encodeChat("You are a helpful assistant",
                "Write a story about a robot.", false);
        Engine e = new Engine(metal, tl, gguf, 8192);
        e.prefill(ids, true);
        int t = Sampler.argmax(e.logits, e.nVocab);
        int warm = 10, n = 150;
        long wall0 = System.nanoTime();
        e.decodeEncodeNanos = 0; e.decodeWaitNanos = 0; e.decodeGpuMs = 0;
        for (int s = 0; s < warm + n; s++) {
            if (s == warm) { wall0 = System.nanoTime(); e.decodeEncodeNanos = 0; e.decodeWaitNanos = 0; e.decodeGpuMs = 0; }
            e.decodeSplit(t, ids.size() + s, true);
            t = Sampler.argmax(e.logits, e.nVocab);
        }
        double wallMs = (System.nanoTime() - wall0) / 1e6 / n;
        System.out.printf("n=%d  wall/token=%.2fms  (%.1f t/s)%n", n, wallMs, 1000 / wallMs);
        System.out.printf("  encode (CPU launch encoding, partially hidden): %.2fms%n", e.decodeEncodeNanos / 1e6 / n);
        System.out.printf("  wait-from-commit-to-completion:                 %.2fms%n", e.decodeWaitNanos / 1e6 / n);
        System.out.printf("  GPU busy (both command buffers, Metal tstamp):  %.2fms%n", e.decodeGpuMs / n);
        System.out.printf("  => GPU idle / host serial per token:           %.2fms%n", wallMs - e.decodeGpuMs / n);
    }
}
