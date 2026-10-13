package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;
/** Engine-level determinism hammer: alternating prompts with resetState,
 *  greedy decode via decodeSplit; compares token sequences. */
public class SplitHammer {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 0);
        List<Integer> pa = tok.encodeChat("You are a helpful assistant",
                "What is the capital of France? Answer in one word.", false);
        int rounds = Integer.parseInt(a.length > 0 ? a[0] : "12");
        Engine e = new Engine(metal, tl, gguf, 8192);
        boolean split = !Boolean.getBoolean("ds4j.noSplit");
        List<Integer> base = run(e, pa, 44, split);
        List<Integer> base2 = run(e, pa, 44, split);
        if (!base.equals(base2)) System.out.println("WARNING: base run not self-consistent");
        int bad = 0;
        for (int k = 0; k < rounds; k++) {
            List<Integer> pb = tok.encodeChat("You are a helpful assistant",
                    "Unrelated question number " + k + ": what is 2+2?", false);
            run(e, pb, 30, split);
            e.resetState();
            List<Integer> out = run(e, pa, 44, split);
            if (!out.equals(base)) {
                bad++;
                int d = -1;
                for (int i = 0; i < Math.min(out.size(), base.size()); i++)
                    if (!out.get(i).equals(base.get(i))) { d = i; break; }
                System.out.printf("round %d DIFFERS at tok %d (len %d vs %d): base=%s now=%s%n",
                        k, d, base.size(), out.size(),
                        d >= 0 ? tok.decodeToken(base.get(d)) : "?",
                        d >= 0 ? tok.decodeToken(out.get(d)) : "?");
            }
        }
        System.out.println(bad + "/" + rounds + " mismatches (engine-level)");
    }
    static List<Integer> run(Engine e, List<Integer> prompt, int steps, boolean split) {
        e.resetState();
        e.prefill(prompt, true);
        List<Integer> out = new ArrayList<>();
        int t = Sampler.argmax(e.logits, e.nVocab);
        for (int s = 0; s < steps; s++) {
            out.add(t);
            if (split) e.decodeSplit(t, prompt.size() + s, true);
            else { try (Metal.Batch bb = e.begin()) { e.decode(bb, t, prompt.size() + s, true); } }
            t = Sampler.argmax(e.logits, e.nVocab);
        }
        return out;
    }
}
