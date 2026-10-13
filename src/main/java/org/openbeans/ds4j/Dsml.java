package org.openbeans.ds4j;

import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Map;
import java.util.List;

/**
 * The DeepSeek DSML tool-call protocol, ported from ds4-server: prompt
 * instructions, tool-result replay, and the parse of model output into
 * structured calls. Four marker styles are accepted (full DSML, short DSML,
 * plain XML, DSML-with-space V4.1); the canonical prompt teaches the full one.
 */
public final class Dsml {

    // ---- markers (｜ is U+FF5C, as in the C's DS4_DSML) ----
    static final String DSML = "｜DSML｜";
    static final String DSML_SHORT = "DSML｜";
    static final String TC_START = "<" + DSML + "tool_calls>";
    static final String TC_END = "</" + DSML + "tool_calls>";
    static final String INV_START = "<" + DSML + "invoke";
    static final String INV_END = "</" + DSML + "invoke>";
    static final String PAR_START = "<" + DSML + "parameter";
    static final String PAR_END = "</" + DSML + "parameter>";
    static final String TC_START_S = "<" + DSML_SHORT + "tool_calls>";
    static final String TC_END_S = "</" + DSML_SHORT + "tool_calls>";
    static final String INV_START_S = "<" + DSML_SHORT + "invoke";
    static final String INV_END_S = "</" + DSML_SHORT + "invoke>";
    static final String PAR_START_S = "<" + DSML_SHORT + "parameter";
    static final String PAR_END_S = "</" + DSML_SHORT + "parameter>";
    static final String TOOL_RESULT = "</tool_result>";

    // ---- incremental tool-call decode tracker (C: dsml_decode_tracker) ----
    // Decides, per generated token, whether decoding is inside DSML tool-call
    // structure (greedy in the C) or inside sampled payload text. Mirrors
    // dsml_decode_tracker_update from ds4_server.c.

    static final String[][] TRACK_SYNTAXES = {
            // {tool_calls_start, tool_calls_end, invoke_start, invoke_end,
            //  param_start, param_end} — C: dsml_syntaxes[] (deepseek rows)
            {"<｜DSML｜ calls>", "</｜DSML｜ calls>", "<｜DSML｜ invoke", "</｜DSML｜ invoke>",
             "<｜DSML｜ parameter", "</｜DSML｜ parameter>"},
            {TC_START, TC_END, INV_START, INV_END, PAR_START, PAR_END},
            {TC_START_S, TC_END_S, INV_START_S, INV_END_S, PAR_START_S, PAR_END_S},
            {"<tool_calls>", "</tool_calls>", "<invoke", "</invoke>",
             "<parameter", "</parameter>"},
    };

    public static final class Tracker {
        public static final int OUTSIDE = 0, STRUCTURAL = 1, STRING_BODY = 2,
                JSON_STRUCTURAL = 3, JSON_STRING = 4;
        static final int M_SEARCH = 0, M_STRUCTURAL = 1, M_STRING_BODY = 2,
                M_JSON_PARAM = 3, M_DONE = 4;
        int mode = M_SEARCH;
        int decode = OUTSIDE;
        String[] syn;
        int pos;
        boolean jsonInString, jsonEscaped;

        static int maxToolStartLen() {
            int max = 0;
            for (String[] s : TRACK_SYNTAXES) max = Math.max(max, s[0].length());
            return max;
        }

        static int indexOf(String raw, int from, String lit) {
            return raw.indexOf(lit, Math.min(from, raw.length()));
        }

        static boolean fullLit(String raw, int pos, String lit) {
            return raw.startsWith(lit, pos);
        }

        static boolean partialLit(String raw, int pos, String lit) {
            int n = lit.length();
            int rem = raw.length() - pos;
            if (rem <= 0 || rem >= n) return false;
            return raw.regionMatches(pos, lit, 0, rem);
        }

        /** C: dsml_attr_is_string_true for the deepseek syntax (the
         *  " string=\"true\"" attribute of a parameter tag). */
        static boolean attrIsStringTrue(String raw, int tagStart, int tagAfter) {
            int q = raw.indexOf("string", tagStart);
            if (q < 0 || q >= tagAfter) return false;
            int eq = raw.indexOf('=', q);
            if (eq < 0 || eq >= tagAfter) return false;
            int vq1 = raw.indexOf('"', eq);
            if (vq1 < 0 || vq1 >= tagAfter) return false;
            int vq2 = raw.indexOf('"', vq1 + 1);
            if (vq2 < 0 || vq2 >= tagAfter) return false;
            return raw.startsWith("true", vq1 + 1);
        }

        public void reset() {
            mode = M_SEARCH;
            decode = OUTSIDE;
            syn = null;
            jsonInString = false;
            jsonEscaped = false;
        }

        public void update(String raw) {
            int rawLen = raw.length();
            loop:
            while (true) {
                if (mode == M_DONE) { decode = OUTSIDE; return; }

                if (mode == M_SEARCH) {
                    int best = -1;
                    String[] bestSyn = null;
                    int hold = maxToolStartLen();
                    int from = Math.min(pos, rawLen);
                    for (String[] s : TRACK_SYNTAXES) {
                        int i = indexOf(raw, from, s[0]);
                        if (i >= 0 && (best < 0 || i < best)) { best = i; bestSyn = s; }
                    }
                    if (best < 0) {
                        pos = rawLen > hold ? rawLen - hold : 0;
                        decode = OUTSIDE;
                        return;
                    }
                    syn = bestSyn;
                    pos = best + bestSyn[0].length();
                    mode = M_STRUCTURAL;
                    decode = STRUCTURAL;
                }

                if (mode == M_STRING_BODY) {
                    while (pos < rawLen) {
                        if (fullLit(raw, pos, syn[5])) {
                            pos += syn[5].length();
                            mode = M_STRUCTURAL;
                            decode = STRUCTURAL;
                            continue loop;
                        }
                        if (partialLit(raw, pos, syn[5])) {
                            decode = rawLen - pos >= 2 ? STRUCTURAL : STRING_BODY;
                            return;
                        }
                        pos++;
                    }
                    decode = STRING_BODY;
                    return;
                }

                if (mode == M_JSON_PARAM) {
                    while (pos < rawLen) {
                        if (!jsonInString) {
                            if (fullLit(raw, pos, syn[5])) {
                                pos += syn[5].length();
                                mode = M_STRUCTURAL;
                                decode = STRUCTURAL;
                                continue loop;
                            }
                            if (partialLit(raw, pos, syn[5])) {
                                decode = rawLen - pos >= 2 ? STRUCTURAL : JSON_STRUCTURAL;
                                return;
                            }
                        }
                        char c = raw.charAt(pos++);
                        if (jsonInString) {
                            if (jsonEscaped) jsonEscaped = false;
                            else if (c == '\\') jsonEscaped = true;
                            else if (c == '"') jsonInString = false;
                        } else if (c == '"') jsonInString = true;
                    }
                    decode = jsonInString ? JSON_STRING : JSON_STRUCTURAL;
                    return;
                }

                // structural
                while (pos < rawLen && Character.isWhitespace(raw.charAt(pos))) pos++;
                if (pos >= rawLen) { decode = STRUCTURAL; return; }

                if (fullLit(raw, pos, syn[1])) {
                    mode = M_DONE;
                    pos += syn[1].length();
                    decode = OUTSIDE;
                    return;
                }
                if (fullLit(raw, pos, syn[3])) {
                    pos += syn[3].length();
                    continue;
                }
                if (fullLit(raw, pos, syn[2])) {
                    int gt = raw.indexOf('>', pos);
                    if (gt < 0) { decode = STRUCTURAL; return; }
                    pos = gt + 1;
                    continue;
                }
                if (fullLit(raw, pos, syn[4])) {
                    int tagStart = pos;
                    int gt = raw.indexOf('>', pos);
                    if (gt < 0) { decode = STRUCTURAL; return; }
                    int tagAfter = gt + 1;
                    boolean stringValue = attrIsStringTrue(raw, tagStart, tagAfter);
                    pos = tagAfter;
                    if (stringValue) {
                        mode = M_STRING_BODY;
                        decode = STRING_BODY;
                        continue loop;
                    }
                    mode = M_JSON_PARAM;
                    jsonInString = false;
                    jsonEscaped = false;
                    decode = JSON_STRUCTURAL;
                    continue loop;
                }

                // unmatched char: hold pos (C: only GLM advances; the
                // deepseek syntax waits for the next marker, and a partial
                // marker must never be consumed)
                decode = STRUCTURAL;
                return;
            }
        }

        public boolean inToolCall() { return decode != OUTSIDE; }

        /** C: dsml_decode_state_uses_payload_sampling. */
        public boolean usesPayloadSampling() {
            return decode == STRING_BODY || decode == JSON_STRING;
        }
    }

    public static final class ToolCall {
        public String id;
        public String name;
        public String arguments;   // JSON object text
    }

    /** Parse result: content before the tool block + the structured calls. */
    public static final class Parsed {
        public final String content;
        public final String rawToolText;   // exact block as sampled (replay fidelity)
        public final List<ToolCall> calls;
        Parsed(String content, String rawToolText, List<ToolCall> calls) {
            this.content = content;
            this.rawToolText = rawToolText;
            this.calls = calls;
        }
        public boolean hasCalls() { return !calls.isEmpty(); }
    }

    // ---- prompt instructions (C: append_tools_prompt_text, non-v41 branch) ----

    public static String toolsPrompt(String toolSchemas) {
        if (toolSchemas == null || toolSchemas.isEmpty()) return "";
        return "## Tools\n\n"
            + "You have access to a set of tools to help answer the user question. "
            + "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
            + "<｜DSML｜tool_calls>\n"
            + "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
            + "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
            + "...\n"
            + "</｜DSML｜invoke>\n"
            + "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
            + "...\n"
            + "</｜DSML｜invoke>\n"
            + "</｜DSML｜tool_calls>\n\n"
            + "String parameters should be specified as raw text and set `string=\"true\"`. "
            + "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. "
            + "Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. "
            + "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
            + "When thinking mode is enabled, finish reasoning with </think> before any tool calls or final response.\n\n"
            + "Otherwise, output directly after </think> with tool calls or final response.\n\n"
                        + "### Available Tool Schemas\n\n"
            + toolSchemas
            + "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. "
            + "Use the exact parameter names from the schemas.";
    }

    /** One raw JSON schema per line (C: append_raw_json_line semantics). */
    public static String schemasFromTools(List<Object> tools) {
        StringBuilder sb = new StringBuilder();
        for (Object t : tools) {
            if (!(t instanceof Map)) continue;
            Map<?, ?> m = (Map<?, ?>) t;
            Object fn = m.get("function");
            Object schema = fn instanceof Map ? fn : m;
            String line = Json.write(schema);
            if (sb.length() > 0) sb.append('\n');
            sb.append(line);
        }
        return sb.toString();
    }

    // ---- replay rendering ----

    /** Assistant turn replay: the raw sampled block when we have it (KV-cache
     *  fidelity), else a canonical reconstruction. */
    public static String renderCalls(List<ToolCall> calls, String rawToolText) {
        if (calls == null || calls.isEmpty()) return "";
        if (rawToolText != null && !rawToolText.isEmpty()) return rawToolText;
        StringBuilder sb = new StringBuilder("\n\n").append(TC_START).append('\n');
        for (ToolCall tc : calls) {
            sb.append(INV_START).append(" name=\"")
              .append(escapeAttr(tc.name)).append("\">\n");
            if (!appendArgumentsFromJson(sb, tc.arguments)) {
                sb.append(PAR_START).append(" name=\"arguments\" string=\"true\">")
                  .append(escapeParamBody(tc.arguments)).append(PAR_END).append('\n');
            }
            sb.append(INV_END).append('\n');
        }
        sb.append(TC_END);
        return sb.toString();
    }

    /** JSON object -> one parameter tag per member; strings as string="true",
     *  other values as their JSON spelling (C: append_dsml_arguments_from_json).
     *  Returns false when the arguments are not a JSON object (fallback used). */
    static boolean appendArgumentsFromJson(StringBuilder b, String json) {
        Object parsed;
        try { parsed = Json.parse(json); } catch (Exception e) { return false; }
        if (!(parsed instanceof Map)) return false;
        Map<?, ?> m = (Map<?, ?>) parsed;
        if (m.isEmpty() && !json.trim().equals("{}")) return false;
        for (Map.Entry<?, ?> e : m.entrySet()) {
            Object v = e.getValue();
            boolean isString = v instanceof String;
            b.append(PAR_START).append(" name=\"").append(escapeAttr(String.valueOf(e.getKey())))
             .append("\" string=\"").append(isString ? "true" : "false").append("\">");
            if (isString) {
                b.append(escapeParamBody((String) v));
            } else if (v == null) {
                b.append("null");
            } else {
                b.append(escapeParamBody(Json.write(v)));
            }
            b.append(PAR_END).append('\n');
        }
        return true;
    }

    /** Tool-result body: data passthrough, escaping only the closing sentinel
     *  (C: append_tool_result_text). */
    public static String escapeToolResult(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            if (needsEscapeAt(s, i, TOOL_RESULT)) {
                sb.append(s.charAt(i) == '<' ? "&lt;" : "&amp;");
            } else {
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();
    }

    // ---- output parse (C: parse_deepseek_generated_message_ex) ----

    /** Parses post-thinking text. Returns content plus structured calls. */
    public static Parsed parse(String text) {
        if (text == null) text = "";
        // earliest tool-call start across all styles (prefer the block after
        // any "\n\n" separator; the C trims that whitespace from content)
        int start = -1, style = 0;
        int[][] cands = {
                {idx(text, "\n\n" + TC_START), 0}, {idx(text, TC_START), 0},
                {idx(text, "\n\n" + TC_START_S), 2}, {idx(text, TC_START_S), 2},
                {idx(text, "\n\n<tool_calls>"), 1}, {idx(text, "<tool_calls>"), 1},
        };
        for (int[] c : cands) {
            if (c[0] >= 0 && (start < 0 || c[0] < start)) { start = c[0]; style = c[1]; }
        }
        if (start < 0) return new Parsed(text, "", new ArrayList<>());

        String tcS = style == 1 ? "<tool_calls>" : style == 2 ? TC_START_S : TC_START;
        String tcE = style == 1 ? "</tool_calls>" : style == 2 ? TC_END_S : TC_END;
        String invS = style == 1 ? "<invoke" : style == 2 ? INV_START_S : INV_START;
        String invE = style == 1 ? "</invoke>" : style == 2 ? INV_END_S : INV_END;
        String parS = style == 1 ? "<parameter" : style == 2 ? PAR_START_S : PAR_START;
        String parE = style == 1 ? "</parameter>" : style == 2 ? PAR_END_S : PAR_END;

        String content = text.substring(0, start).stripTrailing();

        int p = text.indexOf(tcS, start);
        if (p < 0) return new Parsed(text, "", new ArrayList<>());
        p += tcS.length();
        List<ToolCall> calls = new ArrayList<>();
        StringBuilder args = null;
        String name = null;
        for (;;) {
            p = skipWs(text, p);
            if (text.startsWith(tcE, p)) {
                String raw = text.substring(start, p + tcE.length());
                return new Parsed(content, raw, calls);
            }
            if (!text.startsWith(invS, p)) return new Parsed(content, "", calls); // malformed: keep content, drop calls
            int tagEnd = text.indexOf('>', p);
            if (tagEnd < 0) return new Parsed(content, "", calls);
            String tag = text.substring(p, tagEnd + 1);
            name = attr(tag, "name");
            if (name == null) return new Parsed(content, "", calls);
            p = tagEnd + 1;
            args = new StringBuilder();
            while (true) {
                p = skipWs(text, p);
                if (text.startsWith(invE, p)) { p += invE.length(); break; }
                if (!text.startsWith(parS, p)) return new Parsed(content, "", calls);
                tagEnd = text.indexOf('>', p);
                if (tagEnd < 0) return new Parsed(content, "", calls);
                tag = text.substring(p, tagEnd + 1);
                String pname = attr(tag, "name");
                String isString = attr(tag, "string");
                if (pname == null) return new Parsed(content, "", calls);
                int valueStart = tagEnd + 1;
                // loose nested parameters: a non-string param whose body starts
                // with another parameter tag (C: dsml_parse_nested_params_object)
                if (!"true".equals(isString) && text.startsWith(parS, skipWs(text, valueStart))) {
                    StringBuilder nested = new StringBuilder();
                    int np = parseNested(text, valueStart, parS, parE, nested);
                    if (np < 0) return new Parsed(content, "", calls);
                    addArg(args, pname, nested.length() > 0 ? nested.toString() : "{}", false);
                    p = skipWs(text, np);
                    if (text.startsWith(parE, p)) p += parE.length();
                    continue;
                }
                int valueEnd = text.indexOf(parE, valueStart);
                if (valueEnd < 0) return new Parsed(content, "", calls);
                String rawValue = text.substring(valueStart, valueEnd);
                boolean str = isString == null || isString.equals("true");
                String value = str ? unescapeToolText(rawValue, parE) : rawValue;
                addArg(args, pname, value, str);
                p = valueEnd + parE.length();
            }
            ToolCall tc = new ToolCall();
            tc.id = randomId();
            tc.name = name;
            tc.arguments = "{" + args + "}";
            calls.add(tc);
        }
    }

    /** Loose nested-parameter recovery: {"k":v,...} built from child tags. */
    static int parseNested(String text, int p, String parS, String parE, StringBuilder out) {
        p = skipWs(text, p);
        if (!text.startsWith(parS, p)) return -1;
        for (;;) {
            int tagEnd = text.indexOf('>', p);
            if (tagEnd < 0) return -1;
            String tag = text.substring(p, tagEnd + 1);
            String pname = attr(tag, "name");
            String isString = attr(tag, "string");
            if (pname == null) return -1;
            int valueStart = tagEnd + 1;
            if (!"true".equals(isString) && text.startsWith(parS, skipWs(text, valueStart))) {
                StringBuilder nested = new StringBuilder();
                int np = parseNested(text, valueStart, parS, parE, nested);
                if (np < 0) return -1;
                addArg(out, pname, nested.length() > 0 ? nested.toString() : "{}", false);
                p = skipWs(text, np);
            } else {
                int valueEnd = text.indexOf(parE, valueStart);
                if (valueEnd < 0) return -1;
                boolean str = isString == null || isString.equals("true");
                String value = str
                        ? unescapeToolText(text.substring(valueStart, valueEnd), parE)
                        : text.substring(valueStart, valueEnd);
                addArg(out, pname, value, str);
                p = valueEnd + parE.length();
            }
            p = skipWs(text, p);
            if (text.startsWith(parE, p)) return p + parE.length();
            if (!text.startsWith(parS, p)) return -1;
        }
    }

    static void addArg(StringBuilder args, String name, String value, boolean isString) {
        // C: append_json_arg_pair — compact separators (response args JSON)
        if (args.length() > 0) args.append(',');
        args.append(Json.write(name)).append(':');
        if (isString) {
            args.append(Json.write(value));
        } else {
            String min = Json.minify(value);
            args.append(min.isEmpty() ? "null" : min);
        }
    }

    // ---- streaming holdback (C: text_stream_safe_limit) ----

    /** Content that may be streamed to the client without leaking a possible
     *  tool-call marker: everything before the earliest marker, minus trailing
     *  whitespace and any partial '<'-marker tail while not final. */
    public static int streamSafeLimit(String raw, int start, boolean hasTools, boolean finalFlush) {
        int len = raw.length();
        if (len <= start) return len;
        if (!hasTools) return len;
        int limit = len;
        int tool = findAnyToolStart(raw, start);
        if (tool >= 0) return trimWs(raw, start, tool);
        if (!finalFlush) {
            while (limit > start && Character.isWhitespace(raw.charAt(limit - 1))) limit--;
            int scan = Math.max(start, len - 80);
            for (int i = len; i > scan; i--) {
                if (raw.charAt(i - 1) == '<') {
                    if (i - 1 < limit) limit = i - 1;
                    break;
                }
            }
            limit = trimWs(raw, start, limit);
        }
        return limit;
    }

    public static int findAnyToolStart(String s, int from) {
        int best = -1;
        for (String m : new String[]{TC_START, TC_START_S, "<tool_calls>"}) {
            int i = s.indexOf(m, from);
            if (i >= 0 && (best < 0 || i < best)) best = i;
        }
        return best;
    }

    // ---- helpers ----

    static int idx(String s, String needle) { return s.indexOf(needle); }

    static int skipWs(String s, int p) {
        while (p < s.length() && Character.isWhitespace(s.charAt(p))) p++;
        return p;
    }

    static int trimWs(String s, int start, int limit) {
        while (limit > start && Character.isWhitespace(s.charAt(limit - 1))) limit--;
        return limit;
    }

    /** XML-attribute extraction with entity decoding (C: dsml_attr). */
    static String attr(String tag, String name) {
        int p = tag.indexOf(name + "=\"");
        if (p < 0) return null;
        p += name.length() + 2;
        int q = tag.indexOf('"', p);
        if (q < 0) return null;
        return unescapeXml(tag.substring(p, q));
    }

    static String unescapeXml(String s) {
        if (s.indexOf('&') < 0) return s;
        return s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", "\"").replace("&apos;", "'");
    }

    static String escapeAttr(String s) {
        return s == null ? "" : s.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace("\"", "&quot;");
    }

    static String escapeParamBody(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (needsEscapeAt(s, i, PAR_END)) {
                sb.append(s.charAt(i) == '<' ? "&lt;" : "&amp;");
            } else {
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();
    }

    /** The escaped-close spelling of the sentinel, e.g. "&amp;lt;/｜DSML｜parameter>"
     *  (C: ds4_tool_text_escaped_close), or the sentinel itself. */
    static boolean escapedCloseAt(String s, int i, String end) {
        if (s.charAt(i) != '&') return false;
        int p = i + 1;
        while (s.startsWith("amp;", p)) p += 4;
        if (!s.startsWith("lt;", p)) return false;
        return s.regionMatches(p + 3, end, 1, end.length() - 1);
    }

    /** C: ds4_tool_text_needs_escape — the sentinel or its escaped spelling. */
    static boolean needsEscapeAt(String s, int i, String end) {
        return s.startsWith(end, i) || escapedCloseAt(s, i, end);
    }

    /** Narrow unescape: only escaped-close spellings of the sentinel (C:
     *  ds4_tool_text_unescape); ordinary entities pass through unchanged. */
    static String unescapeToolText(String s, String end) {
        if (s.indexOf('&') < 0) return s;
        StringBuilder out = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            if (escapedCloseAt(s, i, end)) {
                if (s.startsWith("&amp;", i)) { out.append('&'); i += 4; }
                else { out.append('<'); i += 3; }
            } else {
                out.append(s.charAt(i));
            }
        }
        return out.toString();
    }

    static final SecureRandom RNG = new SecureRandom();

    public static String randomId() {
        byte[] b = new byte[16];
        RNG.nextBytes(b);
        StringBuilder sb = new StringBuilder("call_");
        for (byte x : b) {
            sb.append(Character.forDigit((x >> 4) & 0xf, 16));
            sb.append(Character.forDigit(x & 0xf, 16));
        }
        return sb.toString();
    }
}