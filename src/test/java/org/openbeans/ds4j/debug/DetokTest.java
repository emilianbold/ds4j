package org.openbeans.ds4j.debug;

import org.openbeans.ds4j.*;
import java.nio.file.*;
import java.util.*;

public class DetokTest {
    public static void main(String[] a) throws Exception {
        Metal metal = Metal.open(MetalSources.source(),
                "DS4_METAL_HAS_TENSOR=1\nDS4_METAL_HC_STABLE=1\nDS4_METAL_NORM_RSQRT_DISABLE=1");
        Tl tl = new Tl(metal, System.getProperty("ds4j.tunedDir"));
        Gguf gguf = Gguf.open(System.getProperty("ds4j.model", "ds4flash.gguf"));
        Tokenizer tok = new Tokenizer(gguf, 0);

        // unit: partial sequences held back, complete ones emitted
        byte[][] cases = {
            {0x41, (byte)0xf0, (byte)0x9f, (byte)0x98},          // A + 3/4 of emoji -> hold 3
            {(byte)0xf0, (byte)0x9f, (byte)0x98, (byte)0x82},     // full emoji -> all
            {0x61, (byte)0xf0},                                    // a + lead -> hold 1
            {(byte)0xe2, (byte)0xac},                              // 2-byte lead+1 -> hold 2
        };
        int[] want = {1, 4, 1, 0};
        for (int k = 0; k < cases.length; k++) {
            int got = Detok.safeUtf8Prefix(cases[k], cases[k].length);
            System.out.printf("case %d: got=%d want=%d %s%n", k, got, want[k],
                    got == want[k] ? "OK" : "FAIL");
        }

        // e2e: emoji text token-by-token vs whole-string decode
        for (String text : new String[]{
                "the standard punchline is still \"At the bottom.\" \uD83D\uDE04",
                "\uD83E\uDD17 hello \u4f60\u597d \uD83C\uDF89 caf\u00e9 na\u00efve"}) {
            List<Integer> ids = tok.encodeChatText(text);
            Detok d = new Detok(tok);
            StringBuilder sb = new StringBuilder();
            for (int id : ids) sb.append(d.push(id));
            sb.append(d.flush());
            String whole = tok.decodeTokens(ids);
            System.out.println("ids=" + ids.size() + " match=" + sb.toString().equals(whole));
            if (!sb.toString().equals(whole)) {
                System.out.println("  streamed: " + sb);
                System.out.println("  whole   : " + whole);
            }
        }
        // token-boundary check: find a token pair that splits an emoji
        List<Integer> ids = tok.encodeChatText("a \uD83D\uDE04 b");
        for (int id : ids) System.out.println("  tok " + id + " bytes=" + java.util.Arrays.toString(tok.tokenBytes(id)));
    }
}
