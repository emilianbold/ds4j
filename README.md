# ds4j — pure-Java DeepSeek V4 Flash inference for Apple Silicon

A Java reimplementation of the C `ds4-server` from
[antirez/ds4](https://github.com/antirez/ds4) (DwarfStar): TileLang Metal kernels,
an OpenAI/Anthropic-compatible HTTP API, DSML tool calls, and streaming.

The JDK is the only dependency, no native shim and no third-party libraries.

## Requirements

- **macOS, Apple Silicon** (Metal; developed on an M5 Max)
- **JDK 25+** (`java.lang.foreign` FFM)
- **Maven 3.9+** (build only)
- **The model**: `ds4flash.gguf` — DeepSeek V4 Flash Q2 (IQ2XXS experts,
  ~81 GB, the quant recommended for 96–128 GB RAM Macs). Download from
  Hugging Face — [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf),
  file `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`,
  then pass `--model /path/to/ds4flash.gguf`
  (without `--model` the server looks for `ds4flash.gguf` in the working
  directory).

## Build & run

    mvn package            # -> target/ds4j.jar

    java --enable-native-access=ALL-UNNAMED -jar target/ds4j.jar \
         --model /path/to/ds4flash.gguf --port 8080

One-shot CLI instead of the server:

    java --enable-native-access=ALL-UNNAMED -cp target/ds4j.jar org.openbeans.ds4j.Main -p "..."

## Server

    --port N --host H      listen (default 127.0.0.1:8000)
    --ctx N                context size (default 32768)
    -m, --model FILE       model gguf; -h/--help for the full usage

OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`
(streaming + tools), `/v1/completions`, plus Anthropic `/v1/messages`.
Seeded requests reproduce the C engine's sampled tokens exactly.
