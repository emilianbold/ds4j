package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;
public class GenProbe {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        List<Integer> ids = new ArrayList<>();
        for (String p : Files.readString(Path.of(a[0])).trim().split("\\s+"))
            ids.add(Integer.parseInt(p));
        int ctx = Integer.parseInt(a[1]);
        int steps = Integer.parseInt(a[2]);
        Engine e = new Engine(metal, tl, gguf, ctx);
        e.prefill(ids, true);
        int tok = Sampler.argmax(e.logits, e.nVocab);
        StringBuilder sb = new StringBuilder();
        Tokenizer tok_ = new Tokenizer(gguf, e.nVocab);
        for (int s = 0; s < steps; s++) {
            sb.append(tok_.decodeToken(tok));
            try (Metal.Batch bb = metal.begin()) {
                e.decode(bb, tok, ids.size() + s, true);
            }
            tok = Sampler.argmax(e.logits, e.nVocab);
        }
        System.out.println("ctx=" + ctx + " n=" + ids.size() + " gen: " + sb);
    }
}
