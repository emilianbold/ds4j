package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;
public class SplitCheck {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 0);
        List<Integer> ids = tok.encodeChat("You are a helpful assistant",
                "What is the capital of France? Answer in one word.", false);
        int steps = 30;
        // path 1: single-buffer decode
        Engine e1 = new Engine(metal, tl, gguf, 8192);
        e1.prefill(ids, true);
        int t1 = Sampler.argmax(e1.logits, e1.nVocab);
        StringBuilder s1 = new StringBuilder();
        for (int s = 0; s < steps; s++) {
            s1.append(tok.decodeToken(t1));
            try (Metal.Batch bb = metal.begin()) { e1.decode(bb, t1, ids.size() + s, true); }
            t1 = Sampler.argmax(e1.logits, e1.nVocab);
        }
        // path 2: split decode
        Engine e2 = new Engine(metal, tl, gguf, 8192);
        e2.prefill(ids, true);
        int t2 = Sampler.argmax(e2.logits, e2.nVocab);
        StringBuilder s2 = new StringBuilder();
        for (int s = 0; s < steps; s++) {
            s2.append(tok.decodeToken(t2));
            e2.decodeSplit(t2, ids.size() + s, true);
            t2 = Sampler.argmax(e2.logits, e2.nVocab);
        }
        System.out.println("single: " + s1);
        System.out.println("split : " + s2);
        System.out.println("identical: " + s1.toString().equals(s2.toString()));
    }
}
