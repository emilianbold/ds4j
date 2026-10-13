package org.openbeans.ds4j;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * GPT-2 byte-level BPE with the JoyAI pre-tokenizer split, ported from
 * ds4.c (bpe_tokenize_text / bpe_emit_piece / vocab_load). Loads
 * tokenizer.ggml.tokens + merges from the GGUF metadata.
 */
public final class Tokenizer {
    public final String[] tokens;   // id -> token string (byte-encoded space)
    public final List<String> specials;  // rendered-chat split markers, longest first
    public final Map<String, Integer> tokenToId = new HashMap<>();
    public final Map<String, Integer> mergeRank = new HashMap<>();
    public final int bosId, eosId, userId, assistantId, thinkStartId, thinkEndId;
    public final int nVocab;

    // GPT-2 byte<->codepoint tables
    static final int[] BYTE_TO_CP = new int[256];
    static final Map<Integer, Integer> CP_TO_BYTE = new HashMap<>();
    static {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac)
                    || (b >= 0xae && b <= 0xff);
            BYTE_TO_CP[b] = printable ? b : 256 + n++;
        }
        for (int b = 0; b < 256; b++) CP_TO_BYTE.put(BYTE_TO_CP[b], b);
    }

    public Tokenizer(Gguf gguf, int nVocab) {
        this.nVocab = nVocab;
        Object[] toks = (Object[]) gguf.meta.get("tokenizer.ggml.tokens");
        Object[] merges = (Object[]) gguf.meta.get("tokenizer.ggml.merges");
        if (toks == null || merges == null)
            throw new RuntimeException("missing tokenizer.ggml metadata");
        tokens = new String[toks.length];
        for (int i = 0; i < toks.length; i++) {
            tokens[i] = (String) toks[i];
            tokenToId.put(tokens[i], i);
        }
        for (int i = 0; i < merges.length; i++)
            mergeRank.put((String) merges[i], i);
        bosId = req("<｜begin▁of▁sentence｜>");
        eosId = req("<｜end▁of▁sentence｜>");
        userId = req("<｜User｜>");
        assistantId = req("<｜Assistant｜>");
        thinkStartId = req("<think>");
        thinkEndId = req("</think>");
    
        // special tokens for rendered-chat splitting (C: special_token_at):
        // every vocab marker containing the fullwidth bar, plus the two
        // plain think markers
        java.util.ArrayList<String> sp = new java.util.ArrayList<>();
        for (String tk : tokens)
            if (tk.indexOf('\uff5c') >= 0 || tk == null) sp.add(tk);
        sp.add(tokens[thinkStartId]);
        sp.add(tokens[thinkEndId]);
        sp.sort((a, b) -> b.length() - a.length());
        specials = List.copyOf(sp);
    }

    int req(String s) {
        Integer id = tokenToId.get(s);
        if (id == null) throw new RuntimeException("missing tokenizer token: " + s);
        return id;
    }

    /** Tokenize rendered chat text: spans of plain BPE between special
     *  markers, which become their exact ids (C: tokenize_rendered_chat_vocab). */
    public List<Integer> encodeChatText(String text) {
        List<Integer> out = new ArrayList<>();
        int spanStart = 0, i = 0, n = text.length();
        while (i < n) {
            int matchId = -1, matchLen = 0;
            for (String sp : specials) {
                if (text.regionMatches(i, sp, 0, sp.length())) {
                    matchId = tokenToId.get(sp);
                    matchLen = sp.length();
                    break;
                }
            }
            if (matchLen > 0) {
                if (i > spanStart) out.addAll(encode(text.substring(spanStart, i)));
                out.add(matchId);
                i += matchLen;
                spanStart = i;
            } else {
                i++;
            }
        }
        if (n > spanStart) out.addAll(encode(text.substring(spanStart)));
        return out;
    }

    // ---- byte-level encode/decode ----
    static byte[] byteEncode(String piece) {
        byte[] in = piece.getBytes(StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder(in.length * 2);
        for (byte b : in) sb.appendCodePoint(BYTE_TO_CP[b & 0xff]);
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    public String decodeToken(int id) {
        // token string lives in the GPT-2 byte-encoder space; map each
        // codepoint back to its original byte, then decode the bytes as UTF-8.
        // Raw special tokens (codepoints outside the byte domain) pass through
        // their own UTF-8 bytes.
        return new String(tokenBytes(id), StandardCharsets.UTF_8);
    }

    public String decodeTokens(List<Integer> ids) {
        // concat raw bytes first, then decode: a multi-byte character can
        // straddle a token boundary and per-token decoding would mangle it
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream(64);
        for (int id : ids) out.writeBytes(tokenBytes(id));
        return new String(out.toByteArray(), StandardCharsets.UTF_8);
    }

    /** Raw GPT-2-byte-encoded bytes of a token, mapped back to original bytes. */
    public byte[] tokenBytes(int id) {
        String t = tokens[id];
        byte[] enc = t.getBytes(StandardCharsets.UTF_8);
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream(enc.length);
        int i = 0;
        boolean raw = false;
        while (i < enc.length) {
            int b0 = enc[i] & 0xff;
            int len = 1;
            if ((b0 & 0xe0) == 0xc0) len = 2;
            else if ((b0 & 0xf0) == 0xe0) len = 3;
            else if ((b0 & 0xf8) == 0xf0) len = 4;
            if (i + len > enc.length) len = 1;
            int cp = new String(enc, i, len, StandardCharsets.UTF_8).codePointAt(0);
            i += len;
            Integer ob = CP_TO_BYTE.get(cp);
            if (ob != null) {
                out.write(ob);
            } else {
                // codepoint outside the 256-byte domain: this is a raw
                // (non-byte-encoded) special token (e.g. "｜DSML｜") — its
                // string IS its UTF-8 bytes
                raw = true;
                break;
            }
        }
        return raw ? enc : out.toByteArray();
    }

    /** One turn of a chat conversation. role: system | user | assistant | tool. */
    public static final class ChatMsg {
        public final String role;
        public final String content;
        /** assistant turns: structured calls to replay as DSML */
        public final List<Dsml.ToolCall> calls;
        /** assistant turns: exact DSML block as sampled, for KV-cache fidelity */
        public final String rawToolText;
        public ChatMsg(String role, String content) {
            this(role, content, null, null);
        }
        public ChatMsg(String role, String content, List<Dsml.ToolCall> calls, String rawToolText) {
            this.role = role;
            this.content = content;
            this.calls = calls;
            this.rawToolText = rawToolText;
        }
        public static ChatMsg system(String c) { return new ChatMsg("system", c); }
        public static ChatMsg user(String c) { return new ChatMsg("user", c); }
        public static ChatMsg assistant(String c) { return new ChatMsg("assistant", c); }
        public static ChatMsg assistantToolCalls(List<Dsml.ToolCall> calls, String raw, String content) {
            return new ChatMsg("assistant", content == null ? "" : content, calls, raw);
        }
        public static ChatMsg tool(String c) { return new ChatMsg("tool", c); }
    }

    /**
     * Builds the DeepSeek chat template over a full history:
     * BOS, optional system, then user/assistant turns wrapped in the
     * role markers, opening the assistant slot for generation. Assistant
     * turns close with the think-end marker if think mode was on (the model
     * always enters through the think-open marker when thinking is enabled).
     */
    public List<Integer> encodeChatHistory(List<ChatMsg> history, boolean think) {
        return encodeChatHistory(history, null, think);
    }

    /** Full render (C: render_deepseek_chat_prompt_text): BOS, tool schemas,
     *  system, user/assistant/tool turns with DSML call replay and
     *  <tool_result> blocks, then the open assistant slot. Consecutive
     *  tool results share one user marker. */
    public List<Integer> encodeChatHistory(List<ChatMsg> history, String toolSchemas, boolean think) {
        String toolPrompt = Dsml.toolsPrompt(toolSchemas);
        List<String> systemParts = new ArrayList<>();
        if (!toolPrompt.isEmpty()) systemParts.add(toolPrompt);
        String bos = tokens[bosId], user = tokens[userId], asst = tokens[assistantId];
        String thinkOpen = tokens[thinkStartId], thinkClose = tokens[thinkEndId];
        String eos = tokens[eosId];
        StringBuilder out = new StringBuilder();
        out.append(bos);
        boolean pendingAssistant = false, pendingToolResult = false;
        for (ChatMsg m : history) {
            switch (m.role) {
                case "system" -> {
                    if (m.content != null && !m.content.isEmpty()) systemParts.add(m.content);
                }
                case "user" -> {
                    out.append(user).append(m.content == null ? "" : m.content);
                    pendingAssistant = true;
                    pendingToolResult = false;
                }
                case "tool" -> {
                    if (!pendingToolResult) out.append(user);
                    out.append("<tool_result>")
                       .append(Dsml.escapeToolResult(m.content == null ? "" : m.content))
                       .append("</tool_result>");
                    pendingAssistant = true;
                    pendingToolResult = true;
                }
                case "assistant" -> {
                    if (pendingAssistant) {
                        out.append(asst).append(thinkClose);
                    }
                    if (m.content != null && !m.content.isEmpty()) out.append(m.content);
                    out.append(Dsml.renderCalls(m.calls, m.rawToolText));
                    out.append(eos);
                    pendingAssistant = false;
                    pendingToolResult = false;
                }
                default -> throw new IllegalArgumentException("role: " + m.role);
            }
        }
        out.append(asst).append(think ? thinkOpen : thinkClose);
        if (!systemParts.isEmpty())
            out.insert(bos.length(), String.join("\n\n", systemParts));
        return encodeChatText(out.toString());
    }

    // ---- BPE ----
    int mergeRankOf(String a, String b) {
        Integer r = mergeRank.get(a + " " + b);
        return r == null ? -1 : r;
    }

    List<Integer> bpe(String rawPiece) {
        String enc = new String(byteEncode(rawPiece), StandardCharsets.UTF_8);
        List<String> sym = new ArrayList<>();
        int i = 0;
        while (i < enc.length()) {
            int cp = enc.codePointAt(i);
            int len = Character.charCount(cp);
            sym.add(new String(Character.toChars(cp)));
            i += len;
        }
        while (sym.size() > 1) {
            int bestI = -1, bestRank = Integer.MAX_VALUE;
            for (int j = 0; j + 1 < sym.size(); j++) {
                int r = mergeRankOf(sym.get(j), sym.get(j + 1));
                if (r >= 0 && r < bestRank) { bestRank = r; bestI = j; }
            }
            if (bestI < 0) break;
            sym.set(bestI, sym.get(bestI) + sym.get(bestI + 1));
            sym.remove(bestI + 1);
        }
        List<Integer> out = new ArrayList<>(sym.size());
        for (String s : sym) {
            Integer id = tokenToId.get(s);
            if (id == null)
                throw new RuntimeException("BPE produced unknown symbol: [" + s + "]");
            out.add(id);
        }
        return out;
    }

    // ---- JoyAI pre-tokenizer (port of bpe_tokenize_text) ----
    static boolean digit(byte c) { return c >= '0' && c <= '9'; }
    static boolean alpha(byte c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
    static boolean newline(byte c) { return c == '\n' || c == '\r'; }
    static boolean space(byte c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'
            || c == 0x0b || c == 0x0c; }
    static boolean punctSymbol(byte c) {
        return (c >= '!' && c <= '/') || (c >= ':' && c <= '@')
                || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
    }

    static int nextUtf8(byte[] s, int pos) {
        int b0 = s[pos] & 0xff;
        if (b0 < 0x80) return pos + 1;
        int need = 1;
        if ((b0 & 0xe0) == 0xc0) need = 2;
        else if ((b0 & 0xf0) == 0xe0) need = 3;
        else if ((b0 & 0xf8) == 0xf0) need = 4;
        return Math.min(s.length, pos + need);
    }

    static int codePointAt(byte[] s, int pos) {
        return new String(s, pos, nextUtf8(s, pos) - pos, StandardCharsets.UTF_8)
                .codePointAt(0);
    }

    static boolean cjkAt(byte[] s, int pos) {
        if ((s[pos] & 0xff) < 128) return false;
        int cp = codePointAt(s, pos);
        return (cp >= 0x2E80 && cp <= 0x9FFF)      // CJK radicals..CJK unified
                || (cp >= 0x3040 && cp <= 0x30FF)  // hiragana/katakana
                || (cp >= 0xF900 && cp <= 0xFAFF)
                || (cp >= 0x20000 && cp <= 0x2FA1F);
    }

    static boolean letterLikeAt(byte[] s, int pos) {
        byte c = s[pos];
        if ((c & 0xff) < 128) return alpha(c);
        return true; // non-ASCII non-control treated as letters (see ds4.c)
    }

    static int consumeLetters(byte[] s, int pos) {
        while (pos < s.length && letterLikeAt(s, pos)) pos = nextUtf8(s, pos);
        return pos;
    }

    public List<Integer> encode(String text) {
        List<Integer> out = new ArrayList<>();
        byte[] b = text.getBytes(StandardCharsets.UTF_8);
        int pos = 0;
        while (pos < b.length) {
            int start = pos;
            byte c = b[pos];
            if (digit(c)) {
                int nd = 0;
                while (pos < b.length && digit(b[pos]) && nd < 3) { pos++; nd++; }
            } else if (cjkAt(b, pos)) {
                do { pos = nextUtf8(b, pos); } while (pos < b.length && cjkAt(b, pos));
            } else if (punctSymbol(c) && pos + 1 < b.length && alpha(b[pos + 1])) {
                pos++;
                while (pos < b.length && alpha(b[pos])) pos++;
            } else if (letterLikeAt(b, pos)) {
                pos = consumeLetters(b, pos);
            } else if (!newline(c) && !punctSymbol(c) && pos + 1 < b.length
                    && letterLikeAt(b, pos + 1)) {
                pos++;
                pos = consumeLetters(b, pos);
            } else if (c == ' ' && pos + 1 < b.length && punctSymbol(b[pos + 1])) {
                pos++;
                while (pos < b.length && punctSymbol(b[pos])) pos++;
                while (pos < b.length && newline(b[pos])) pos++;
            } else if (punctSymbol(c)) {
                while (pos < b.length && punctSymbol(b[pos])) pos++;
                while (pos < b.length && newline(b[pos])) pos++;
            } else if (space(c)) {
                int p = pos;
                int lastNewlineEnd = 0;
                while (p < b.length && space(b[p])) {
                    byte sc = b[p++];
                    if (newline(sc)) lastNewlineEnd = p;
                }
                if (lastNewlineEnd > 0) {
                    pos = lastNewlineEnd;
                } else if (p < b.length && p > pos + 1
                        && (letterLikeAt(b, p) || punctSymbol(b[p]))) {
                    pos = p - 1;
                } else {
                    pos = p;
                }
            } else {
                pos = nextUtf8(b, pos);
            }
            if (pos == start) pos = nextUtf8(b, pos);
            String piece = new String(b, start, pos - start, StandardCharsets.UTF_8);
            out.addAll(bpe(piece));
        }
        return out;
    }

    /** Chat prompt: BOS [system] <｜User｜> prompt <｜Assistant｜>  </think> (or think-open). */
    public List<Integer> encodeChat(String system, String prompt, boolean think) {
        List<Integer> out = new ArrayList<>();
        out.add(bosId);
        if (system != null && !system.isEmpty()) out.addAll(encode(system));
        out.add(userId);
        out.addAll(encode(prompt));
        out.add(assistantId);
        out.add(think ? thinkStartId : thinkEndId);
        return out;
    }
}
