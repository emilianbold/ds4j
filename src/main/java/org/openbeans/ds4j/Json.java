package org.openbeans.ds4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON reader/writer for the server (no external deps).
 * Parsing yields Map&lt;String,Object&gt; / List / String / Double / Boolean / null.
 */
public final class Json {
    private Json() {}

    // ---------------- parse ----------------

    public static Object parse(String s) {
        P p = new P(s);
        p.ws();
        Object v = p.value();
        p.ws();
        if (p.i < s.length()) throw new IllegalArgumentException("trailing JSON at " + p.i);
        return v;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> obj(Object v) {
        if (v instanceof Map) return (Map<String, Object>) v;
        throw new IllegalArgumentException("expected JSON object");
    }

    private static final class P {
        final String s;
        int i;
        P(String s) { this.s = s; }
        void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }
        char peek() { if (i >= s.length()) throw new IllegalArgumentException("EOF"); return s.charAt(i); }

        Object value() {
            switch (peek()) {
                case '{': return object();
                case '[': return array();
                case '"': return string();
                case 't': expect("true"); return Boolean.TRUE;
                case 'f': expect("false"); return Boolean.FALSE;
                case 'n': expect("null"); return null;
                default: return number();
            }
        }

        void expect(String lit) {
            if (!s.startsWith(lit, i)) throw new IllegalArgumentException("bad literal at " + i);
            i += lit.length();
        }

        Map<String, Object> object() {
            Map<String, Object> m = new LinkedHashMap<>();
            i++; ws();
            if (peek() == '}') { i++; return m; }
            while (true) {
                String k = string();
                ws();
                if (peek() != ':') throw new IllegalArgumentException("expected : at " + i);
                i++; ws();
                m.put(k, value());
                ws();
                char c = peek();
                if (c == ',') { i++; ws(); }
                else if (c == '}') { i++; return m; }
                else throw new IllegalArgumentException("bad object at " + i);
            }
        }

        List<Object> array() {
            List<Object> l = new ArrayList<>();
            i++; ws();
            if (peek() == ']') { i++; return l; }
            while (true) {
                l.add(value());
                ws();
                char c = peek();
                if (c == ',') { i++; ws(); }
                else if (c == ']') { i++; return l; }
                else throw new IllegalArgumentException("bad array at " + i);
            }
        }

        String string() {
            if (peek() != '"') throw new IllegalArgumentException("expected string at " + i);
            StringBuilder sb = new StringBuilder();
            i++;
            while (true) {
                char c = s.charAt(i++);
                if (c == '"') return sb.toString();
                if (c == '\\') {
                    char e = s.charAt(i++);
                    switch (e) {
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        case '/' -> sb.append('/');
                        case 'b' -> sb.append('\b');
                        case 'f' -> sb.append('\f');
                        case 'n' -> sb.append('\n');
                        case 'r' -> sb.append('\r');
                        case 't' -> sb.append('\t');
                        case 'u' -> {
                            sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                            i += 4;
                        }
                        default -> throw new IllegalArgumentException("bad escape \\" + e);
                    }
                } else {
                    sb.append(c);
                }
            }
        }

        Object number() {
            int j = i;
            if (j < s.length() && (s.charAt(j) == '-' || s.charAt(j) == '+')) j++;
            boolean isDouble = false;
            while (j < s.length()) {
                char c = s.charAt(j);
                if (c >= '0' && c <= '9') j++;
                else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
                    if (c == '.' || c == 'e' || c == 'E') isDouble = true;
                    j++;
                } else break;
            }
            String num = s.substring(i, j);
            i = j;
            if (!isDouble) {
                try { long v = Long.parseLong(num); return v; }
                catch (NumberFormatException ignore) {}
            }
            return Double.parseDouble(num);
        }
    }

    // ---------------- write ----------------

    public static String write(Object v) {
        StringBuilder sb = new StringBuilder();
        writeValue(sb, v);
        return sb.toString();
    }

    static void writeValue(StringBuilder sb, Object v) {
        if (v == null) { sb.append("null"); return; }
        if (v instanceof String s) { writeString(sb, s); return; }
        if (v instanceof Boolean b) { sb.append(b ? "true" : "false"); return; }
        if (v instanceof Double d) {
            if (d == Math.floor(d) && !d.isInfinite() && Math.abs(d) < 1e15)
                sb.append((long) (double) d);
            else sb.append(d);
            return;
        }
        if (v instanceof Number n) { sb.append(n); return; }
        if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                writeString(sb, String.valueOf(e.getKey()));
                sb.append(':');
                writeValue(sb, e.getValue());
            }
            sb.append('}');
            return;
        }
        if (v instanceof List<?> l) {
            sb.append('[');
            boolean first = true;
            for (Object o : l) {
                if (!first) sb.append(',');
                first = false;
                writeValue(sb, o);
            }
            sb.append(']');
            return;
        }
        writeString(sb, String.valueOf(v));
    }

    static void writeString(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
                }
            }
        }
        sb.append('"');
    }

    /** Minified JSON: whitespace outside strings removed (invalid -> ""). */
    public static String minify(String json) {
        StringBuilder sb = new StringBuilder(json.length());
        boolean inStr = false;
        for (int i = 0; i < json.length(); i++) {
            char c = json.charAt(i);
            if (inStr) {
                sb.append(c);
                if (c == '\\') {
                    if (i + 1 < json.length()) sb.append(json.charAt(++i));
                } else if (c == '"') inStr = false;
            } else if (c == '"') {
                inStr = true;
                sb.append(c);
            } else if (!Character.isWhitespace(c)) {
                sb.append(c);
            }
        }
        return inStr ? "" : sb.toString();   // unterminated string: not valid
    }

    // ---------------- typed getters ----------------

    public static String str(Map<String, Object> m, String k, String def) {
        Object v = m.get(k);
        return v instanceof String s ? s : def;
    }

    public static double dbl(Map<String, Object> m, String k, double def) {
        Object v = m.get(k);
        if (v instanceof Number n) return n.doubleValue();
        return def;
    }

    public static int intVal(Map<String, Object> m, String k, int def) {
        Object v = m.get(k);
        if (v instanceof Number n) return (int) Math.round(n.doubleValue());
        return def;
    }

    public static boolean bool(Map<String, Object> m, String k, boolean def) {
        Object v = m.get(k);
        if (v instanceof Boolean b) return b;
        return def;
    }

    @SuppressWarnings("unchecked")
    public static List<Map<String, Object>> objects(List<Object> l) {
        List<Map<String, Object>> out = new ArrayList<>();
        if (l == null) return out;
        for (Object o : l) out.add(obj(o));
        return out;
    }

    @SuppressWarnings("unchecked")
    public static List<Object> list(Map<String, Object> m, String k) {
        Object v = m.get(k);
        return v instanceof List ? (List<Object>) v : List.of();
    }
}