package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;

/** Generate raw-prompt fixtures for boundary testing: space-separated
 *  digits tokenize to exactly one token each, so text "1 2 ... n" has n
 *  tokens. Writes <prefix>.ids and prints the text to stdout for the C's
 *  --raw-prompt. Usage: RawGen <prefix> <n...> */
public class RawGen {
    public static void main(String[] a) throws Exception {
        String src = MetalSources.source();
        Metal metal = Metal.open(src,
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 0);
        for (int ai = 1; ai < a.length; ai++) {
            int n = Integer.parseInt(a[ai]);
            StringBuilder sb = new StringBuilder();
            for (int i = 1; i <= n; i++) {
                if (i > 1) sb.append(' ');
                sb.append("cat");
            }
            List<Integer> ids = tok.encodeChatText(sb.toString());
            if (ids.size() != n)
                throw new IllegalStateException("text for n=" + n + " tokenized to " + ids.size());
            StringBuilder out = new StringBuilder();
            for (int i : ids) out.append(i).append('\n');
            Files.writeString(Path.of(a[0] + "_n" + n + ".ids"), out.toString());
            System.out.println("n=" + n + " text=[" + sb + "]");
        }
    }
}
