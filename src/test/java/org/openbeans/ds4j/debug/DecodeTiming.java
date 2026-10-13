package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;

/** Decode-phase CPU/GPU split: prefill a short prompt, decode N tokens,
 *  report avg encode (Java + dispatch) vs GPU-wait per token. */
public class DecodeTiming {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 0);
        List<Integer> ids = tok.encodeChat("You are a helpful assistant",
                "Write a story about a robot.", false);
        Engine e = new Engine(metal, tl, gguf, 8192);
        System.out.print("compRatio: [");
        for (int i = 0; i < e.nLayer; i++) System.out.print(e.compRatio[i] + (i < e.nLayer - 1 ? "," : ""));
        System.out.println("]");
        e.prefill(ids, true);
        long launchBaseline = Tl.LAUNCHES.get();
        int tokId = Sampler.argmax(e.logits, e.nVocab);
        int warm = 10, n = Integer.parseInt(a.length > 0 ? a[0] : "150");
        for (int s = 0; s < warm + n; s++) {
            if (s == warm) { e.decodeEncodeNanos = 0; e.decodeWaitNanos = 0; Tl.TRACE_COUNTS.clear(); }
            try (Metal.Batch bb = metal.begin()) {
                e.decode(bb, tokId, ids.size() + s, true);
            }
            tokId = Sampler.argmax(e.logits, e.nVocab);
        }
        double enc = e.decodeEncodeNanos / 1e6 / n, wait = e.decodeWaitNanos / 1e6 / n;
        long launches = Tl.LAUNCHES.get();
        System.out.printf("backend=ffm-pure n=%d avg encode=%.2fms  gpu-wait=%.2fms  total=%.2fms  (%.1f t/s)%n",
                n, enc, wait, enc + wait, 1000.0 / (enc + wait));
        System.out.printf("launches/token = %.1f (incl. prefill share: %d total)%n",
                (launches - launchBaseline) / (double) n, launches);
        java.util.List<String[]> rows = new java.util.ArrayList<>();
        Tl.TRACE_COUNTS.forEach((sym, c) -> rows.add(new String[]{sym, String.valueOf(c[0])}));
        rows.sort((x, y) -> Long.compare(Long.parseLong(y[1]), Long.parseLong(x[1])));
        for (String[] r2 : rows)
            System.out.printf("  %-55s %6.2f/token%n", r2[0], Long.parseLong(r2[1]) / (double) n);
    }
}
