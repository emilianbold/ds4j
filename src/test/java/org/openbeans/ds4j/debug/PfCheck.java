package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.util.List;

public final class PfCheck {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, "."); // tuned selection: resource first, repo fallback
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 129280);
        List<Integer> ids = tok.encodeChat("You are a helpful assistant",
                "Say hello and tell me what model you are.", false);
        System.out.println("tokens: " + ids.size());

        // decode-path logits
        Engine e1 = new Engine(metal, tl, gguf, 1024);
        Metal.Buffer lg1 = null;
        for (int pos = 0; pos < ids.size(); pos++) {
            try (Metal.Batch bb = metal.begin()) { lg1 = e1.decode(bb, ids.get(pos), pos, true); }
        }
        int am1 = Sampler.argmax(lg1, e1.nVocab);
        System.out.println("decode argmax: " + am1 + " [" + tok.decodeToken(am1) + "]");

        // prefill logits
        Engine e2 = new Engine(metal, tl, gguf, 1024);
        Metal.Buffer lg2 = e2.prefill(ids, true);
        int am2 = Sampler.argmax(lg2, e2.nVocab);
        System.out.println("prefill argmax: " + am2 + " [" + tok.decodeToken(am2) + "]");

        float md = 0; int ma = -1;
        for (int i = 0; i < e1.nVocab; i++) {
            float d = Math.abs(lg1.getFloat(i * 4L) - lg2.getFloat(i * 4L));
            if (d > md) { md = d; ma = i; }
        }
        System.out.printf("prefill vs decode logits: max diff %.4f at %d (decode=%.3f prefill=%.3f)%n",
                md, ma, lg1.getFloat(ma * 4L), lg2.getFloat(ma * 4L));
    }
}
