package org.openbeans.ds4j;

import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Streaming detokenizer. The BPE decoder maps one token to a fragment of
 * UTF-8 bytes; a multi-byte character can straddle a token boundary, so
 * raw byte output is buffered until a complete character (or the end of
 * the stream) is available. {@code push} returns whatever text is safely
 * complete; {@code flush} drains the remainder.
 */
public final class Detok {
    private final Tokenizer tok;
    private final java.io.ByteArrayOutputStream pending = new java.io.ByteArrayOutputStream(64);
    private int bosId, eosId, thinkStartId, thinkEndId;

    public Detok(Tokenizer tok) {
        this.tok = tok;
        this.bosId = tok.bosId;
        this.eosId = tok.eosId;
        this.thinkStartId = tok.thinkStartId;
        this.thinkEndId = tok.thinkEndId;
    }

    /**
     * Feeds one token id and returns the decoded text that is safely
     * complete after it. Special tokens (BOS/EOS/think markers) are
     * consumed and produce no text.
     */
    public String push(int id) {
        if (id == bosId || id == eosId || id == thinkStartId || id == thinkEndId) return "";
        byte[] bytes = tok.tokenBytes(id);
        pending.writeBytes(bytes);
        return drain(false);
    }

    /** Drains any buffered partial bytes at end of stream. */
    public String flush() {
        return drain(true);
    }

    private String drain(boolean flush) {
        byte[] buf = pending.toByteArray();
        int len = buf.length;
        int safe = flush ? len : safeUtf8Prefix(buf, len);
        if (safe == 0) return "";
        String s = new String(buf, 0, safe, StandardCharsets.UTF_8);
        pending.reset();
        if (safe < len) pending.writeBytes(java.util.Arrays.copyOfRange(buf, safe, len));
        return s;
    }

    /** Length of the longest prefix that ends on a character boundary. */
    public static int safeUtf8Prefix(byte[] b, int len) {
        // forward scan: emit only fully available sequences; an incomplete
        // multi-byte sequence at the tail is held back for the next token
        int i = 0;
        while (i < len) {
            int c = b[i] & 0xff;
            int need;
            if ((c & 0x80) == 0) need = 1;
            else if ((c & 0xe0) == 0xc0) need = 2;
            else if ((c & 0xf0) == 0xe0) need = 3;
            else if ((c & 0xf8) == 0xf0) need = 4;
            else return i; // stray continuation byte: emit as-is (replacement)
            if (i + need > len) return i;
            i += need;
        }
        return len;
    }
}