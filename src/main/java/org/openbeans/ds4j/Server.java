package org.openbeans.ds4j;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

/**
 * OpenAI/Anthropic-compatible server built on the ds4j library API — the
 * moral equivalent of ds4-server, except the transport is the JDK's built-in
 * HTTP server instead of hand-rolled sockets. One live session is kept
 * (C parity): a new request reuses the live KV cache when its rendered
 * prompt extends the live token prefix, otherwise the session is reset and
 * prefilled fresh.
 */
public final class Server {

    final Model model;
    final int ctx;
    final HttpServer http;
    final ExecutorService pool = Executors.newFixedThreadPool(8);
    final Object liveLock = new Object();
    Session liveSession = null;
    final List<Integer> liveTokens = new ArrayList<>();
    final AtomicLong ids = new AtomicLong();
    volatile boolean shuttingDown = false;

    public Server(Model model, int port, String host, int ctx) throws IOException {
        this.model = model;
        this.ctx = ctx;
        this.http = HttpServer.create(new InetSocketAddress(host, port), 64);
        http.setExecutor(pool);
        http.createContext("/v1/models", this::handleModels);
        http.createContext("/v1/chat/completions", this::handleChatCompletions);
        http.createContext("/v1/completions", this::handleCompletions);
        http.createContext("/v1/messages", this::handleMessages);
        http.createContext("/health", ex -> sendJson(ex, 200, Map.of("status", "ok")));
        http.createContext("/", ex -> sendJson(ex, 200, Map.of(
                "server", "ds4j", "endpoints", List.of(
                        "/v1/models", "/v1/chat/completions", "/v1/completions",
                        "/v1/messages", "/health"))));
    }

    int decodePrefill = Integer.getInteger("ds4j.decodePrefill", 0);

    static final String USAGE = """
            Usage: ds4-server [options]  (ds4j)

              -m, --model FILE    model gguf (default: ds4flash.gguf in the working dir)
              -c, --ctx N         context size (default: 32768)
                  --host HOST     listen address (default: 127.0.0.1)
                  --port N        listen port (default: 8000)
              -h, --help          this help

            ds4j-specific options are -Dds4j.* system properties, e.g.
            -Dds4j.prefillChunk=N.
            """;

    public static void main(String[] args) throws Exception {
        int port = 8000;
        String host = "127.0.0.1";
        int ctx = Integer.parseInt(System.getProperty("ds4j.ctx", "32768"));
        String modelPath = System.getProperty("ds4j.model");
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port" -> port = Integer.parseInt(args[++i]);
                case "--host" -> host = args[++i];
                case "--ctx", "-c" -> ctx = Integer.parseInt(args[++i]);
                case "--model", "-m" -> modelPath = args[++i];
                case "-h", "--help" -> { System.out.print(USAGE); return; }
                default -> {
                    System.err.println("ds4-server: unknown option: " + args[i]);
                    System.err.print(USAGE);
                    System.exit(2);
                }
            }
        }
        if (modelPath == null) modelPath = "ds4flash.gguf";
        long t0 = System.nanoTime();
        try (Model m = Model.load(modelPath)) {
            Server s = new Server(m, port, host, ctx);
            s.http.start();
            System.out.printf("[ds4j-server] %s on http://%s:%d/ (ctx %d, model loaded in %.1f s)%n",
                    m.metal.deviceName, host, port, ctx, (System.nanoTime() - t0) / 1e9);
            if (s.decodePrefill > 0)
                System.out.printf("[ds4j-server] decode-prefill mode: new prompts <= %d tokens use the decode path%n",
                        s.decodePrefill);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> s.shutdown()));
            Thread.currentThread().join();
        }
    }

    void shutdown() {
        shuttingDown = true;
        http.stop(0);
        pool.shutdownNow();
    }

    // ================= helpers =================

    static void sendJson(HttpExchange ex, int code, Object v) throws IOException {
        byte[] b = (Json.write(v) + "\n").getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json");
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.sendResponseHeaders(code, b.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(b); }
    }

    static void error(HttpExchange ex, int code, String msg) throws IOException {
        if (msg == null || "null".equals(msg))
            Thread.currentThread().getStackTrace();
        sendJson(ex, code, Map.of("error", Map.of("message", msg, "type", "invalid_request_error")));
    }

    static Map<String, Object> body(HttpExchange ex) throws IOException {
        byte[] b = ex.getRequestBody().readAllBytes();
        if (b.length == 0) throw new IllegalArgumentException("empty body");
        return Json.obj(Json.parse(new String(b, StandardCharsets.UTF_8)));
    }

    String rawBodyOf(HttpExchange ex) throws IOException {
        byte[] b = ex.getRequestBody().readAllBytes();
        return b.length == 0 ? "" : new String(b, StandardCharsets.UTF_8);
    }

    record ParsedRequest(Map<String, Object> map, String raw) {}

    ParsedRequest parseBody(HttpExchange ex) throws IOException {
        byte[] b = ex.getRequestBody().readAllBytes();
        if (b.length == 0) throw new IllegalArgumentException("empty body");
        String raw = new String(b, StandardCharsets.UTF_8);
        System.out.printf("[ds4j-server] %s %s (%d bytes)%n",
                ex.getRequestMethod(), ex.getRequestURI().getPath(), b.length);
        String dump = System.getProperty("ds4j.dumpReq");
        if (dump != null) {
            try { java.nio.file.Files.writeString(java.nio.file.Path.of(dump),
                    raw, java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.APPEND); }
            catch (Exception ignore) {}
        }
        return new ParsedRequest(Json.obj(Json.parse(raw)), raw);
    }

    /**
     * C parity: chat requests default to THINKING ON (the C server's
     * DS4_THINK_HIGH default). Disabling: reasoning_effort "none" or a
     * thinking {"type":"disabled"} object. Enabling explicitly: any other
     * effort level or thinking {"type":"enabled"}.
     */
    public static boolean thinkFor(String modelAlias, Map<String, Object> req) {
        Object th = req.get("thinking");
        if (th instanceof Map<?, ?> m) {
            if ("enabled".equals(m.get("type"))) return true;
            if ("disabled".equals(m.get("type"))) return false;
        }
        Object eff = req.get("reasoning_effort");
        if (eff instanceof String e) {
            if (e.equals("none")) return false;
            return true;
        }
        return true;
    }

    String newId(String prefix) { return prefix + Long.toHexString(System.currentTimeMillis()) + ids.incrementAndGet(); }

    // ================= /v1/models =================

    void handleModels(HttpExchange ex) throws IOException {
        try {
            String path = ex.getRequestURI().getPath();
            if ("GET".equals(ex.getRequestMethod())) {
                if (path.equals("/v1/models") || path.equals("/v1/models/")) {
                    sendJson(ex, 200, Map.of("object", "list", "data", List.of(
                            modelJson("deepseek-v4-flash"), modelJson("deepseek-v4-pro"))));
                } else {
                    sendJson(ex, 200, modelJson(path.substring("/v1/models/".length())));
                }
            } else {
                error(ex, 405, "method not allowed");
            }
        } catch (Exception e) {
            error(ex, 500, String.valueOf(e.getMessage()));
        }
    }

    Map<String, Object> modelJson(String id) {
        int maxCompletion = Math.min(ctx, 8192);
        Map<String, Object> top = new LinkedHashMap<>();
        top.put("context_length", ctx);
        top.put("max_completion_tokens", maxCompletion);
        top.put("is_moderated", false);
        return Map.of("id", id, "object", "model", "created", 1767225600L,
                "owned_by", "ds4j", "context_length", ctx, "top_provider", top);
    }

    // ================= prompt rendering + live-session reuse =================

    /** Renders an OpenAI messages array into the DeepSeek chat template,
     *  injecting tool schemas and replaying tool calls / results (C parity).
     *  ALL system messages are kept (joined into the top system region by
     *  the tokenizer) — an injected reminder must not replace the main
     *  system prompt. */
    public List<Integer> renderChat(Map<String, Object> req, boolean think, String toolSchemas) {
        Tokenizer tok = model.tokenizer;
        List<Tokenizer.ChatMsg> msgs = new ArrayList<>();
        for (Map<String, Object> m : Json.objects(Json.list(req, "messages"))) {
            String role = Json.str(m, "role", "user");
            String content = contentString(m.get("content"));
            switch (role) {
                case "system", "developer" -> {
                    if (content != null && !content.isEmpty())
                        msgs.add(Tokenizer.ChatMsg.system(content));
                }
                case "user" -> msgs.add(Tokenizer.ChatMsg.user(content));
                case "assistant" -> {
                    List<Dsml.ToolCall> calls = openaiToolCalls(m.get("tool_calls"));
                    if (calls != null && !calls.isEmpty())
                        msgs.add(Tokenizer.ChatMsg.assistantToolCalls(calls, null, content));
                    else
                        msgs.add(Tokenizer.ChatMsg.assistant(content));
                }
                case "tool" -> msgs.add(Tokenizer.ChatMsg.tool(content));
                default -> throw new IllegalArgumentException("role: " + role);
            }
        }
        return tok.encodeChatHistory(msgs, toolSchemas, think);
    }

    /** OpenAI tool_calls array -> structured calls (arguments JSON passthrough). */
    @SuppressWarnings("unchecked")
    public static List<Dsml.ToolCall> openaiToolCalls(Object v) {
        if (!(v instanceof List<?> l) || l.isEmpty()) return null;
        List<Dsml.ToolCall> out = new ArrayList<>();
        for (Object o : l) {
            if (!(o instanceof Map<?, ?>)) continue;
            Map<?, ?> tc = (Map<?, ?>) o;
            Map<?, ?> fn = tc.get("function") instanceof Map<?, ?> f ? f : Map.of();
            Dsml.ToolCall call = new Dsml.ToolCall();
            call.id = String.valueOf(tc.get("id") == null ? "" : tc.get("id"));
            call.name = String.valueOf(fn.get("name") == null ? "" : fn.get("name"));
            Object args = fn.get("arguments");
            call.arguments = args instanceof String s ? (s.isEmpty() ? "{}" : s)
                    : Json.write(args == null ? Map.of() : args);
            out.add(call);
        }
        return out;
    }

    /** tools + tool_choice -> the schema text injected into the system region
     *  (C: parse_tools_value; tool_choice "none" disables). The schema lines
     *  keep the request's RAW bytes — the C slices the "function" value
     *  straight out of the body, preserving its exact spacing. */
    public static String toolSchemasFor(Map<String, Object> req, String rawBody) {
        Object choice = req.get("tool_choice");
        if ("none".equals(choice)) return "";
        List<Object> tools = Json.list(req, "tools");
        if (tools.isEmpty()) return "";
        String raw = rawToolSchemas(rawBody);
        return raw != null ? raw : Dsml.schemasFromTools(tools);
    }

    /** Raw JSON scanner: the "tools" array, each element unwrapped to its
     *  top-level "function" value (or kept whole), joined one per line with
     *  the body's exact bytes preserved. Returns null when not parseable. */
    public static String rawToolSchemas(String body) {
        if (body == null) return null;
        try {
            int[] p = {0};
            skipWs(body, p);
            if (p[0] >= body.length() || body.charAt(p[0]) != '{') return null;
            p[0]++;
            while (true) {
                skipWs(body, p);
                if (p[0] >= body.length()) return null;
                char c = body.charAt(p[0]);
                if (c == '}') return null;
                if (c != '"') return null;
                String key = rawString(body, p);
                skipWs(body, p);
                if (p[0] >= body.length() || body.charAt(p[0]) != ':') return null;
                p[0]++;
                skipWs(body, p);
                if ("tools".equals(key)) {
                    if (p[0] >= body.length() || body.charAt(p[0]) != '[') return null;
                    p[0]++;
                    StringBuilder sb = new StringBuilder();
                    while (true) {
                        skipWs(body, p);
                        if (p[0] >= body.length()) return null;
                        c = body.charAt(p[0]);
                        if (c == ']') break;
                        int start = p[0];
                        if (rawValueEnd(body, p) < 0) return null;
                        String schema = unwrapFunction(body, start, p[0]);
                        String line = schema != null ? schema : body.substring(start, p[0]);
                        if (sb.length() > 0) sb.append('\n');
                        sb.append(line);
                        skipWs(body, p);
                        if (p[0] < body.length() && body.charAt(p[0]) == ',') p[0]++;
                    }
                    return sb.length() == 0 ? null : sb.toString();
                }
                if (rawValueEnd(body, p) < 0) return null;
                skipWs(body, p);
                if (p[0] < body.length() && body.charAt(p[0]) == ',') p[0]++;
            }
        } catch (Exception e) {
            return null;
        }
    }

    static void skipWs(String s, int[] p) {
        while (p[0] < s.length() && Character.isWhitespace(s.charAt(p[0]))) p[0]++;
    }

    /** Parses a JSON string at p, returns its decoded value, advances past it. */
    static String rawString(String s, int[] p) {
        if (s.charAt(p[0]) != '"') throw new IllegalArgumentException("expected string");
        StringBuilder sb = new StringBuilder();
        p[0]++;
        while (true) {
            char c = s.charAt(p[0]++);
            if (c == '"') return sb.toString();
            if (c == '\\') {
                char e = s.charAt(p[0]++);
                switch (e) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> { sb.append((char) Integer.parseInt(s.substring(p[0], p[0] + 4), 16)); p[0] += 4; }
                    default -> throw new IllegalArgumentException("bad escape");
                }
            } else {
                sb.append(c);
            }
        }
    }

    /** Advances p past one complete JSON value at p (ws already skipped).
     *  Returns the end index, or -1 on error. */
    static int rawValueEnd(String s, int[] p) {
        int start = p[0];
        if (start >= s.length()) return -1;
        char c = s.charAt(start);
        if (c == '"') { try { rawString(s, p); return p[0]; } catch (Exception e) { return -1; } }
        if (c == '{' || c == '[') {
            char open = c, close = c == '{' ? '}' : ']';
            int depth = 0;
            while (p[0] < s.length()) {
                char x = s.charAt(p[0]);
                if (x == '"') { try { rawString(s, p); continue; } catch (Exception e) { return -1; } }
                if (x == open) depth++;
                else if (x == close) { depth--; if (depth == 0) { p[0]++; return p[0]; } }
                p[0]++;
            }
            return -1;
        }
        // literal: number/true/false/null
        while (p[0] < s.length()) {
            char x = s.charAt(p[0]);
            if (x == ',' || x == '}' || x == ']' || Character.isWhitespace(x)) break;
            p[0]++;
        }
        return p[0] > start ? p[0] : -1;
    }

    /** Within body[start,end) — a JSON object — the raw slice of the
     *  top-level "function" value, or null. */
    static String unwrapFunction(String s, int start, int end) {
        int[] p = {start};
        skipWs(s, p);
        if (p[0] >= end || s.charAt(p[0]) != '{') return null;
        p[0]++;
        while (true) {
            skipWs(s, p);
            if (p[0] >= end) return null;
            char c = s.charAt(p[0]);
            if (c == '}') return null;
            if (c != '"') return null;
            String key = rawString(s, p);
            skipWs(s, p);
            if (p[0] >= end || s.charAt(p[0]) != ':') return null;
            p[0]++;
            skipWs(s, p);
            int vStart = p[0];
            if (rawValueEnd(s, p) < 0 || p[0] > end) return null;
            if ("function".equals(key))
                return s.substring(vStart, p[0]);
            skipWs(s, p);
            if (p[0] < end && s.charAt(p[0]) == ',') p[0]++;
        }
    }

    /** Anthropic content blocks: tool_use entries on assistant messages. */
    static List<Dsml.ToolCall> anthropicToolUses(Object c) {
        if (!(c instanceof List<?> l)) return null;
        List<Dsml.ToolCall> out = new ArrayList<>();
        for (Object o : l) {
            if (!(o instanceof Map<?, ?> m)) continue;
            if (!"tool_use".equals(m.get("type"))) continue;
            Dsml.ToolCall call = new Dsml.ToolCall();
            call.id = String.valueOf(m.get("id") == null ? "" : m.get("id"));
            call.name = String.valueOf(m.get("name") == null ? "" : m.get("name"));
            Object input = m.get("input");
            call.arguments = input == null ? "{}" : Json.write(input);
            out.add(call);
        }
        return out;
    }

    public static String contentString(Object c) {
        if (c instanceof String s) return s;
        if (c instanceof List<?> l) {
            StringBuilder sb = new StringBuilder();
            for (Object part : l)
                if (part instanceof Map<?, ?> m && "text".equals(m.get("type")))
                    sb.append(m.get("text"));
            return sb.toString();
        }
        return "";
    }

    GenerateParams params(Map<String, Object> req, boolean think) {
        GenerateParams p = GenerateParams.defaults();
        p.think = think;
        if (req.containsKey("temperature")) p.temperature = (float) Json.dbl(req, "temperature", 1.0);
        if (req.containsKey("top_p")) p.topP = (float) Json.dbl(req, "top_p", 1.0);
        if (req.containsKey("min_p")) p.minP = (float) Json.dbl(req, "min_p", 0.05);
        if (req.containsKey("top_k")) p.topK = Json.intVal(req, "top_k", 0);
        p.maxTokens = Json.intVal(req, "max_tokens",
                Json.intVal(req, "max_completion_tokens", 0));
        p.seed = (long) Json.dbl(req, "seed", 42);
        Object stop = req.get("stop");
        List<String> stops = new ArrayList<>();
        if (stop instanceof String s) stops.add(s);
        else if (stop instanceof List<?> l) for (Object o : l) stops.add(String.valueOf(o));
        p.stop = stops;
        return p;
    }

    /**
     * Aligns the live session with the wanted prompt tokens: continues from
     * the common prefix when possible, otherwise resets and starts fresh.
     * Returns the token list to feed.
     */
    List<Integer> alignLive(List<Integer> want) {
        if (liveSession != null) {
            int k = commonPrefix(liveTokens, want);
            boolean exact = k == liveTokens.size() && liveSession.contextUsed() == k;
            if (exact && !want.isEmpty()) return want.subList(k, want.size());
            liveSession.reset();
            liveSession.setDecodePrefill(decodePrefill);
            liveTokens.clear();
        } else {
            liveSession = model.newSession(ctx);
            liveSession.setDecodePrefill(decodePrefill);
        }
        return want;
    }

    static int commonPrefix(List<Integer> a, List<Integer> b) {
        int n = Math.min(a.size(), b.size());
        for (int i = 0; i < n; i++)
            if (!a.get(i).equals(b.get(i))) return i;
        return n;
    }

    /** Runs generation on the live session and records exactly the tokens
     *  that landed in the cache — including on abort, so an interrupted
     *  stream never desyncs the prefix ledger from the KV state. */
    Completion runLive(List<Integer> prompt, GenerateParams p, Session.Listener l) {
        try {
            return liveSession.generate(prompt, p, l);
        } finally {
            liveTokens.addAll(liveSession.lastFedIds);
        }
    }

    // ================= /v1/chat/completions =================

    void handleChatCompletions(HttpExchange ex) throws IOException {
        try {
            if (!"POST".equals(ex.getRequestMethod())) { error(ex, 405, "POST only"); return; }
            ParsedRequest pr = parseBody(ex);
            long rt0 = System.nanoTime();
            Map<String, Object> req = pr.map();
            boolean think = thinkFor(Json.str(req, "model", null), req);
            String toolSchemas = toolSchemasFor(req, pr.raw());
            List<Integer> want = renderChat(req, think, toolSchemas);
            long rt1 = System.nanoTime();
            String logPrompt = System.getProperty("ds4j.logPrompt");
            if (logPrompt != null) {
                try {
                    java.nio.file.Files.writeString(java.nio.file.Path.of(logPrompt),
                            model.tokenizer.decodeTokens(want));
                } catch (Exception e) {
                    System.err.println("[ds4j-server] prompt log failed: " + e);
                }
            }
            GenerateParams p = params(req, think);
            p.hasTools = !toolSchemas.isEmpty();
            boolean stream = Json.bool(req, "stream", false);
            if (want.size() + 1 > ctx) {
                sendJson(ex, 400, Map.of("error", Map.of("message",
                        "prompt of " + want.size() + " tokens exceeds context " + ctx,
                        "type", "invalid_request_error", "code", "context_length_exceeded")));
                return;
            }
            if (Boolean.getBoolean("ds4j.reqTiming"))
                System.err.printf("[ds4j-timing] tokens=%d parse+render=%.0fms%n",
                        want.size(), (rt1 - rt0) / 1e6);
            synchronized (liveLock) {
                List<Integer> prompt = alignLive(want);
                if (stream) streamChat(ex, prompt, p, Json.str(req, "model", "deepseek-v4-flash"),
                        !toolSchemas.isEmpty(),
                        Json.bool(Json.obj(req.getOrDefault("stream_options", Map.of())), "include_usage", false));
                else nonStreamChat(ex, prompt, p, Json.str(req, "model", "deepseek-v4-flash"));
            }
        } catch (Session.AbortedException ignore) {
            // client went away mid-stream
        } catch (Exception e) {
            System.err.println("[ds4j-server] handler exception:");
            e.printStackTrace();
            try { error(ex, 500, String.valueOf(e.getMessage())); } catch (IOException ignore) {}
        }
    }

    void nonStreamChat(HttpExchange ex, List<Integer> prompt, GenerateParams p, String modelAlias)
            throws IOException {
        StringBuilder think = new StringBuilder(), text = new StringBuilder();
        Completion c = runLive(prompt, p, new Session.Listener() {
            @Override public void onThinking(String d) { think.append(d); }
            @Override public void onText(String d) { text.append(d); }
        });
        Dsml.Parsed parsed = Dsml.parse(c.text);
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        if (think.length() > 0) message.put("reasoning_content", think.toString());
        message.put("content", parsed.content);
        if (parsed.hasCalls()) message.put("tool_calls", openAiToolCallArray(parsed.calls));
        String finish = switch (c.finishReason) {
            case "length", "context" -> "length";
            case "abort" -> "stop";
            default -> parsed.hasCalls() ? "tool_calls" : "stop";
        };
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("finish_reason", finish);
        choice.put("logprobs", null);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", newId("chatcmpl-"));
        out.put("object", "chat.completion");
        out.put("created", System.currentTimeMillis() / 1000);
        out.put("model", modelAlias);
        out.put("choices", List.of(choice));
        out.put("usage", usage(c));
        sendJson(ex, 200, out);
    }

    static List<Object> openAiToolCallArray(List<Dsml.ToolCall> calls) {
        List<Object> out = new ArrayList<>();
        for (int i = 0; i < calls.size(); i++) {
            Dsml.ToolCall tc = calls.get(i);
            Map<String, Object> fn = new LinkedHashMap<>();
            fn.put("name", tc.name);
            fn.put("arguments", tc.arguments == null || tc.arguments.isEmpty() ? "{}" : tc.arguments);
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("index", i);
            entry.put("id", tc.id == null || tc.id.isEmpty() ? Dsml.randomId() : tc.id);
            entry.put("type", "function");
            entry.put("function", fn);
            out.add(entry);
        }
        return out;
    }

    Map<String, Object> usage(Completion c) {
        Map<String, Object> u = new LinkedHashMap<>();
        u.put("prompt_tokens", c.promptTokens);
        u.put("completion_tokens", c.tokens.size());
        u.put("total_tokens", c.promptTokens + c.tokens.size());
        return u;
    }

    /** SSE writer with client-disconnect detection. */
    static final class Sse {
        final HttpExchange ex;
        OutputStream os;
        boolean failed = false;
        Sse(HttpExchange ex) { this.ex = ex; }
        void start() throws IOException {
            ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
            ex.getResponseHeaders().set("Cache-Control", "no-cache");
            ex.getResponseHeaders().set("Connection", "keep-alive");
            ex.sendResponseHeaders(200, 0);
            os = ex.getResponseBody();
        }
        void data(String json) {
            if (failed) return;
            try {
                os.write(("data: " + json + "\n\n").getBytes(StandardCharsets.UTF_8));
                os.flush();
            } catch (IOException e) {
                failed = true;
            }
        }
        void done() { done(true); }
        void done(boolean openaiDone) {
            if (failed) return;
            try {
                if (openaiDone) os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                os.flush();
            } catch (IOException ignore) {
            }
        }
        void close() {
            try { if (os != null) os.close(); } catch (IOException ignore) {}
        }
    }

    void streamChat(HttpExchange ex, List<Integer> prompt, GenerateParams p,
                    String modelAlias, boolean hasTools, boolean includeUsage) throws IOException {
        Sse sse = new Sse(ex);
        sse.start();
        String id = newId("chatcmpl-");
        long created = System.currentTimeMillis() / 1000;
        sse.data(chunk(id, created, modelAlias, "{\"role\":\"assistant\"}", null));
        StringBuilder buf = new StringBuilder();
        int[] emitted = {0};
        Session.Listener l = new Session.Listener() {
            @Override public void onThinking(String d) {
                sse.data(chunk(id, created, modelAlias,
                        "{\"reasoning_content\":" + Json.write(d) + "}", null));
            }
            @Override public void onText(String d) {
                buf.append(d);
                if (!hasTools) {
                    sse.data(chunk(id, created, modelAlias,
                            "{\"content\":" + Json.write(d) + "}", null));
                    return;
                }
                // hold back a possible DSML marker split across deltas
                int lim = Dsml.streamSafeLimit(buf.toString(), emitted[0], true, false);
                if (lim > emitted[0]) {
                    String safe = buf.substring(emitted[0], lim);
                    emitted[0] = lim;
                    sse.data(chunk(id, created, modelAlias,
                            "{\"content\":" + Json.write(safe) + "}", null));
                }
            }
        };
        Completion c;
        try {
            c = runLive(prompt, p, l);
        } catch (Session.AbortedException e) {
            sse.done();
            sse.close();
            return;
        }
        if (sse.failed) { liveSession.abort(); sse.close(); return; }
        String text = buf.toString();
        Dsml.Parsed parsed = Dsml.parse(text);
        if (hasTools && parsed.content.length() > emitted[0]
                && parsed.content.startsWith(text.substring(0, Math.min(emitted[0], text.length())))) {
            // trailing plain content after the last safe cut (no tool block)
            String tail = parsed.content.substring(emitted[0]);
            if (!tail.isEmpty())
                sse.data(chunk(id, created, modelAlias, "{\"content\":" + Json.write(tail) + "}", null));
        } else if (hasTools && !parsed.hasCalls() && text.length() > emitted[0]) {
            String tail = text.substring(emitted[0]);
            if (!tail.isEmpty())
                sse.data(chunk(id, created, modelAlias, "{\"content\":" + Json.write(tail) + "}", null));
        }
        String finish = switch (c.finishReason) {
            case "length", "context" -> "length";
            case "abort" -> "stop";
            default -> parsed.hasCalls() ? "tool_calls" : "stop";
        };
        if (parsed.hasCalls()) {
            for (int i = 0; i < parsed.calls.size(); i++) {
                Dsml.ToolCall tc = parsed.calls.get(i);
                Map<String, Object> fn = new LinkedHashMap<>();
                fn.put("name", tc.name);
                fn.put("arguments", tc.arguments == null ? "{}" : tc.arguments);
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("index", i);
                entry.put("id", tc.id);
                entry.put("type", "function");
                entry.put("function", fn);
                String delta = "{\"tool_calls\":" + Json.write(List.of(entry)) + "}";
                sse.data(chunk(id, created, modelAlias, delta, null));
            }
        }
        sse.data(chunk(id, created, modelAlias, "{}", finish));
        if (includeUsage) {
            Map<String, Object> u = new LinkedHashMap<>();
            u.put("id", id);
            u.put("object", "chat.completion.chunk");
            u.put("created", created);
            u.put("model", modelAlias);
            u.put("choices", List.of());
            u.put("usage", usage(c));
            sse.data(Json.write(u));
        }
        sse.done();
        sse.close();
    }

    static String chunk(String id, long created, String modelAlias, String deltaJson, String finish) {
        StringBuilder sb = new StringBuilder("{\"id\":\"").append(id)
                .append("\",\"object\":\"chat.completion.chunk\",\"created\":").append(created)
                .append(",\"model\":").append(Json.write(modelAlias))
                .append(",\"choices\":[{\"index\":0,\"delta\":").append(deltaJson)
                .append(",\"finish_reason\":");
        if (finish != null) sb.append(Json.write(finish)); else sb.append("null");
        sb.append("}]}");
        return sb.toString();
    }

    // ================= /v1/completions =================

    void handleCompletions(HttpExchange ex) throws IOException {
        try {
            if (!"POST".equals(ex.getRequestMethod())) { error(ex, 405, "POST only"); return; }
            Map<String, Object> req = body(ex);
            Object po = req.get("prompt");
            String promptText;
            if (po instanceof String s) promptText = s;
            else if (po instanceof List<?> l && !l.isEmpty()) promptText = String.valueOf(l.get(0));
            else { error(ex, 400, "missing prompt"); return; }
            boolean think = thinkFor(Json.str(req, "model", null), req);
            List<Integer> want = model.tokenizer.encodeChatHistory(
                    List.of(Tokenizer.ChatMsg.user(promptText)), think);
            GenerateParams p = params(req, think);
            boolean stream = Json.bool(req, "stream", false);
            synchronized (liveLock) {
                List<Integer> prompt = alignLive(want);
                if (stream) streamText(ex, prompt, p, Json.str(req, "model", "deepseek-v4-flash"));
                else nonStreamText(ex, prompt, p, Json.str(req, "model", "deepseek-v4-flash"));
            }
        } catch (Session.AbortedException ignore) {
        } catch (Exception e) {
            try { error(ex, 500, String.valueOf(e.getMessage())); } catch (IOException ignore) {}
        }
    }

    void nonStreamText(HttpExchange ex, List<Integer> prompt, GenerateParams p, String modelAlias)
            throws IOException {
        Completion c = runLive(prompt, p, null);
        String finish = switch (c.finishReason) {
            case "length", "context" -> "length";
            default -> "stop";
        };
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", newId("cmpl-"));
        out.put("object", "text_completion");
        out.put("created", System.currentTimeMillis() / 1000);
        out.put("model", modelAlias);
        out.put("choices", List.of(Map.of("index", 0, "text", c.text, "finish_reason", finish)));
        out.put("usage", usage(c));
        sendJson(ex, 200, out);
    }

    void streamText(HttpExchange ex, List<Integer> prompt, GenerateParams p, String modelAlias)
            throws IOException {
        Sse sse = new Sse(ex);
        sse.start();
        String id = newId("cmpl-");
        long created = System.currentTimeMillis() / 1000;
        Completion c;
        try {
            c = runLive(prompt, p, new Session.Listener() {
                @Override public void onText(String d) {
                    sse.data(textChunk(id, created, modelAlias, d, null));
                }
            });
        } catch (Session.AbortedException e) {
            sse.done();
            sse.close();
            return;
        }
        if (sse.failed) { liveSession.abort(); sse.close(); return; }
        String finish = switch (c.finishReason) {
            case "length", "context" -> "length";
            default -> "stop";
        };
        sse.data(textChunk(id, created, modelAlias, null, finish));
        sse.done();
        sse.close();
    }

    static String textChunk(String id, long created, String modelAlias, String text, String finish) {
        StringBuilder sb = new StringBuilder("{\"id\":\"").append(id)
                .append("\",\"object\":\"text_completion\",\"created\":").append(created)
                .append(",\"model\":").append(Json.write(modelAlias))
                .append(",\"choices\":[{\"index\":0,\"text\":");
        sb.append(text == null ? "\"\"" : Json.write(text));
        sb.append(",\"finish_reason\":");
        if (finish != null) sb.append(Json.write(finish)); else sb.append("null");
        sb.append("}]}");
        return sb.toString();
    }

    // ================= /v1/messages (Anthropic) =================

    void handleMessages(HttpExchange ex) throws IOException {
        try {
            if (!"POST".equals(ex.getRequestMethod())) { error(ex, 405, "POST only"); return; }
            ParsedRequest prA = parseBody(ex);
            Map<String, Object> req = prA.map();
            String rawBodyA = prA.raw();
            boolean think = thinkFor(Json.str(req, "model", null), req);
            Tokenizer tok = model.tokenizer;
            List<Tokenizer.ChatMsg> msgs = new ArrayList<>();
            String system = Json.str(req, "system", null);
            if (system != null && !system.isEmpty()) msgs.add(Tokenizer.ChatMsg.system(system));
            List<Object> tools = Json.list(req, "tools");
            String toolSchemas = tools.isEmpty() ? "" : Dsml.schemasFromTools(tools);  // anthropic: converted schema
            for (Map<String, Object> m : Json.objects(Json.list(req, "messages"))) {
                String role = Json.str(m, "role", "user");
                String content = contentString(m.get("content"));
                List<Dsml.ToolCall> calls = anthropicToolUses(m.get("content"));
                if ("assistant".equals(role) && calls != null && !calls.isEmpty())
                    msgs.add(Tokenizer.ChatMsg.assistantToolCalls(calls, null, content));
                else
                    msgs.add(new Tokenizer.ChatMsg(role, content));
            }
            List<Integer> want = tok.encodeChatHistory(msgs, toolSchemas, think);
            GenerateParams p = params(req, think);
            p.hasTools = !toolSchemas.isEmpty();
            p.maxTokens = Json.intVal(req, "max_tokens", 0); // required by the API
            if (want.size() + 1 > ctx) {
                sendJson(ex, 400, Map.of("type", "error", "error", Map.of("type",
                        "invalid_request_error", "message", "prompt exceeds context " + ctx)));
                return;
            }
            boolean stream = Json.bool(req, "stream", false);
            boolean hasTools = !toolSchemas.isEmpty();
            synchronized (liveLock) {
                List<Integer> prompt = alignLive(want);
                if (stream) streamAnthropic(ex, prompt, p, Json.str(req, "model", "deepseek-v4-flash"), hasTools);
                else nonStreamAnthropic(ex, prompt, p, Json.str(req, "model", "deepseek-v4-flash"), hasTools);
            }
        } catch (Session.AbortedException ignore) {
        } catch (Exception e) {
            try { error(ex, 500, String.valueOf(e.getMessage())); } catch (IOException ignore) {}
        }
    }

    void nonStreamAnthropic(HttpExchange ex, List<Integer> prompt, GenerateParams p, String modelAlias,
                            boolean hasTools) throws IOException {
        StringBuilder think = new StringBuilder(), text = new StringBuilder();
        Completion c = runLive(prompt, p, new Session.Listener() {
            @Override public void onThinking(String d) { think.append(d); }
            @Override public void onText(String d) { text.append(d); }
        });
        Dsml.Parsed parsed = Dsml.parse(c.text);
        List<Object> content = new ArrayList<>();
        if (think.length() > 0)
            content.add(Map.of("type", "thinking", "thinking", think.toString()));
        content.add(Map.of("type", "text", "text", parsed.content));
        if (parsed.hasCalls()) {
            for (Dsml.ToolCall tc : parsed.calls) {
                Map<String, Object> block = new LinkedHashMap<>();
                block.put("type", "tool_use");
                block.put("id", tc.id.startsWith("call_") ? "toolu_" + tc.id.substring(5) : tc.id);
                block.put("name", tc.name);
                Object input;
                try { input = Json.parse(tc.arguments); }
                catch (Exception e) { input = Map.of(); }
                block.put("input", input);
                content.add(block);
            }
        }
        String stopReason = switch (c.finishReason) {
            case "length", "context" -> "max_tokens";
            default -> parsed.hasCalls() ? "tool_use" : "end_turn";
        };
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", newId("msg_"));
        out.put("type", "message");
        out.put("role", "assistant");
        out.put("model", modelAlias);
        out.put("content", content);
        out.put("stop_reason", stopReason);
        out.put("stop_sequence", null);
        out.put("usage", Map.of("input_tokens", c.promptTokens, "output_tokens", c.tokens.size()));
        sendJson(ex, 200, out);
    }

    void streamAnthropic(HttpExchange ex, List<Integer> prompt, GenerateParams p, String modelAlias,
                         boolean hasTools) throws IOException {
        Sse sse = new Sse(ex);
        sse.start();
        String id = newId("msg_");
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("id", id);
        message.put("type", "message");
        message.put("role", "assistant");
        message.put("model", modelAlias);
        message.put("content", List.of());
        message.put("stop_reason", null);
        message.put("stop_sequence", null);
        message.put("usage", Map.of("input_tokens", prompt.size(), "output_tokens", 0));
        Map<String, Object> msgStart = new LinkedHashMap<>();
        msgStart.put("type", "message_start");
        msgStart.put("message", message);
        sse.data(Json.write(msgStart));
        boolean[] inThink = {p.think};
        int[] blockIdx = {0};
        String[] openBlock = {null};
        Completion c;
        Session.Listener l = new Session.Listener() {
            void openBlock(String type) {
                if (openBlock[0] != null) return;
                blockIdx[0]++;
                openBlock[0] = type;
                Map<String, Object> ev = new LinkedHashMap<>();
                ev.put("type", "content_block_start");
                ev.put("index", blockIdx[0] - 1);
                ev.put("content_block", "thinking".equals(type)
                        ? Map.of("type", "thinking", "thinking", "")
                        : Map.of("type", "text", "text", ""));
                sse.data(Json.write(ev));
            }
            @Override public void onThinking(String d) {
                openBlock("thinking");
                Map<String, Object> ev = new LinkedHashMap<>();
                ev.put("type", "content_block_delta");
                ev.put("index", blockIdx[0] - 1);
                ev.put("delta", Map.of("type", "thinking_delta", "thinking", d));
                sse.data(Json.write(ev));
            }
            @Override public void onText(String d) {
                if (inThink[0]) {
                    inThink[0] = false;
                    if (openBlock[0] != null) {
                        Map<String, Object> ev = new LinkedHashMap<>();
                        ev.put("type", "content_block_stop");
                        ev.put("index", blockIdx[0] - 1);
                        sse.data(Json.write(ev));
                        openBlock[0] = null;
                    }
                }
                openBlock("text");
                Map<String, Object> ev = new LinkedHashMap<>();
                ev.put("type", "content_block_delta");
                ev.put("index", blockIdx[0] - 1);
                ev.put("delta", Map.of("type", "text_delta", "text", d));
                sse.data(Json.write(ev));
            }
        };
        try {
            c = runLive(prompt, p, l);
        } catch (Session.AbortedException e) {
            sse.done(false);
            sse.close();
            return;
        }
        if (sse.failed) { liveSession.abort(); sse.close(); return; }
        if (openBlock[0] != null) {
            sse.data(Json.write(Map.of("type", "content_block_stop", "index", blockIdx[0] - 1)));
        }
        String stopReason = switch (c.finishReason) {
            case "length", "context" -> "max_tokens";
            default -> "end_turn";
        };
        Map<String, Object> md = new LinkedHashMap<>();
        md.put("stop_reason", stopReason);
        md.put("stop_sequence", null);
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("type", "message_delta");
        delta.put("delta", md);
        delta.put("usage", Map.of("output_tokens", c.tokens.size()));
        sse.data(Json.write(delta));
        sse.data(Json.write(Map.of("type", "message_stop")));
        sse.done(false);
        sse.close();
    }
}