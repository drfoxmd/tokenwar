# ⚔️ TokenWar

**Compare LLM responses side-by-side in your terminal, then let an AI judge score them.**

TokenWar sends the same prompt to multiple LLM models via any OpenAI-compatible endpoint, displays their responses in a split-pane TUI, and runs an LLM-as-judge evaluation scoring each response on accuracy, helpfulness, clarity, creativity, and conciseness.

```
┌─────────────────────┬──────────────────────┬─────────────────────┐
│ claude-sonnet-4     │ gpt-4o               │ grok-3              │
│                     │                      │                     │
│ The Rust ownership  │ Rust's ownership     │ In Rust, ownership  │
│ system ensures      │ model is a set of    │ is the core concept │
│ memory safety       │ rules that the       │ that makes memory   │
│ without a garbage   │ compiler checks at   │ safe without GC...  │
│ collector...        │ compile time...      │                     │
│                     │                      │                     │
├─────────────────────┴───────────┬──────────┴─────────────────────┤
│ gemini-2.5-flash                │ llama-3.1-70b                  │
│                                 │                                │
│ Ownership in Rust is a          │ Rust uses an ownership model   │
│ discipline enforced by the      │ where each value has exactly   │
│ compiler that governs how       │ one owner at a time...         │
│ memory is managed...            │                                │
│                                 │                                │
└─────────────────────────────────┴────────────────────────────────┘
```

After all responses arrive, the judge scores them:

```
=== Scoreboard ===
1. claude-sonnet-4 - 42.0/50
2. gemini-2.5-flash - 40.5/50
3. gpt-4o - 39.0/50
4. grok-3 - 38.5/50
5. llama-3.1-70b - 37.0/50

=== Details ===

claude-sonnet-4:
  Accuracy: 9.0 (Correct and precise explanation of ownership rules)
  Helpfulness: 8.5 (Directly addresses the question with practical examples)
  Clarity: 8.5 (Well-structured with clear progression of concepts)
  Creativity: 8.0 (Novel analogy comparing ownership to real-world lending)
  Conciseness: 8.0 (Thorough but not verbose)

gpt-4o:
  Accuracy: 8.5 (Accurate coverage of core concepts)
  Helpfulness: 8.0 (Good overview but fewer practical examples)
  ...
```

## Why TokenWar?

### When it's better than just using Claude or ChatGPT

| Use Case | Why TokenWar Wins |
|----------|---------------|
| **Evaluating models for your use case** | See how multiple models handle *your* actual prompts, not benchmarks |
| **Reducing bias in model selection** | An independent judge scores responses — not your gut feeling |
| **Catching hallucinations** | If 4 models agree and 1 doesn't, you've found a hallucination |
| **Prompt engineering** | Instantly see how different models interpret the same prompt |
| **Choosing a model for production** | Real response quality + latency data, not marketing claims |
| **Creative work** | Compare writing styles, get multiple angles on the same topic |
| **Factual research** | Cross-reference answers across models for higher confidence |
| **Cost optimization** | If a cheaper model scores comparably, you've found your winner |

**Example:** You're building a customer support bot. You write 10 representative prompts, run them through TokenWar, and discover that for *your specific domain*, Gemini outperforms GPT-4o while costing less. You'd never know this from public benchmarks.

### When you should just use Claude or ChatGPT

| Situation | Why TokenWar is Overkill |
|-----------|----------------------|
| **Quick one-off questions** | You just need an answer, not a comparison |
| **Conversational/multi-turn chat** | TokenWar is single-turn only — no follow-ups |
| **You already know your preferred model** | No need to compare if you're happy |
| **Cost-sensitive usage** | TokenWar calls N models + a judge = (N+1)x the cost of one model |
| **Image/audio/video tasks** | TokenWar is text-only |
| **You need tool use or function calling** | TokenWar sends plain prompts, no tool schemas |

## Features

- **⚡ Concurrent API calls** — All models queried simultaneously via tokio
- **📺 Terminal UI** — Split-pane ratatui display showing responses as they stream in
- **🏆 LLM-as-judge scoring** — Automated evaluation on 5 criteria (1-10 scale each, 50 max)
- **🔌 Any model, one endpoint** — Works with LiteLLM, OpenRouter, Ollama, or any OpenAI-compatible API
- **📡 Streaming mode** — Watch responses arrive token-by-token with `--stream`
- **📋 Plain text mode** — `--no-tui` for piping output or CI/automation
- **📊 JSON output** — `--json` for machine-readable results with latency data
- **⏱️ Latency tracking** — Per-model response time in milliseconds
- **🔧 Dynamic model list** — Add or remove models by editing one env var, no code changes
- **💪 Fault tolerant** — One model failing doesn't kill the others

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- An OpenAI-compatible API endpoint with at least 2 models

### Build

```bash
git clone https://github.com/drfoxmd/tokenwar.git
cd tokenwar
cargo build --release
```

The binary will be at `target/release/tokenwar`.

## Proxy Setup

TokenWar talks to a single OpenAI-compatible endpoint. You need a proxy that routes to multiple providers. Pick one:

### Option A: LiteLLM (self-hosted, recommended)

LiteLLM gives you one API for 100+ models with zero token markup. Best for homelabbers.

```bash
# Install
pipx install 'litellm[proxy]'

# Create config (litellm_config.yaml)
cat > litellm_config.yaml << 'EOF'
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gemini-2.5-flash
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY

  - model_name: claude-sonnet-4
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: sk-tokenwar-local
EOF

# Start the proxy
litellm --config litellm_config.yaml --port 4000
```

Or with Docker:

```bash
docker run -d --name litellm \
  -p 4000:4000 \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml
```

### Option B: OpenRouter (hosted, zero setup)

No self-hosting required. One API key, 200+ models, small per-token markup.

1. Get an API key at [openrouter.ai](https://openrouter.ai)
2. Set `BASE_URL=https://openrouter.ai/api/v1` in your `.env`

### Option C: Ollama (fully local)

For comparing local models with zero API costs:

```bash
ollama serve  # starts on localhost:11434
ollama pull llama3.1
ollama pull mistral
```

Set `BASE_URL=http://localhost:11434/v1` and `API_KEY=ollama` in your `.env`.

## Configuration

Copy the example env file and fill in your details:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Base URL for the OpenAI-compatible proxy
BASE_URL=http://localhost:4000/v1
API_KEY=sk-tokenwar-local

# Comma-separated list of model names to compare (min 2)
MODELS=gpt-4o,gpt-4o-mini,gemini-2.5-flash,claude-sonnet-4

# Judge model (used to score the responses)
JUDGE_MODEL=gpt-4o

# Optional: separate judge endpoint/key
# JUDGE_BASE_URL=https://api.openai.com/v1
# JUDGE_API_KEY=sk-xxx

# Optional: per-model overrides (if a model needs a different endpoint)
# MODEL_0_BASE_URL=https://api.openai.com/v1
# MODEL_0_API_KEY=sk-xxx
# MODEL_0_NAME=GPT-4o (direct)
```

> **Model names** must match what your proxy expects. For LiteLLM, these are the `model_name` values in your config. For OpenRouter, use their model IDs (e.g. `openai/gpt-4o`).

## Usage

### Basic

```bash
# Pass prompt as argument
tokenwar "Explain the difference between TCP and UDP"

# Pipe from stdin
echo "Write a haiku about Rust" | tokenwar

# From a file
tokenwar < prompt.txt
```

### Options

```bash
# Stream responses token-by-token in the TUI
tokenwar --stream "What is quantum computing?"

# Plain text output (no TUI, good for scripts/CI)
tokenwar --no-tui "Compare REST vs GraphQL"

# JSON output (machine-readable, includes latency per model)
tokenwar --json "Compare REST vs GraphQL"

# Custom timeout (default: 60s)
tokenwar --timeout-secs 120 "Write a detailed essay on climate change"

# Combine flags
tokenwar --stream --timeout-secs 90 "Explain monads to a 5-year-old"
```

### JSON Output

The `--json` flag outputs structured JSON for programmatic consumption:

```json
{
  "prompt": "What is 2+2?",
  "providers": [
    {
      "name": "gpt-4o",
      "model": "gpt-4o",
      "response_text": "2 + 2 = 4.",
      "error": null,
      "latency_ms": 1234
    },
    {
      "name": "gemini-2.5-flash",
      "model": "gemini-2.5-flash",
      "response_text": "The answer is 4.",
      "error": null,
      "latency_ms": 987
    }
  ],
  "scores": { "scores": [...] },
  "metadata": {
    "timestamp": 1738492800,
    "timeout_secs": 60,
    "stream": false
  }
}
```

### TUI Controls

| Key | Action |
|-----|--------|
| `q` | Quit early (skips waiting for remaining responses) |

The TUI automatically exits once all models have responded, then displays the judge scoreboard.

## Architecture

```
                    ┌─────────────────────────────┐
          prompt    │ OpenAI-compatible endpoint   │
       ┌───────────▶│ (LiteLLM / OpenRouter / ...) │────┐
       │            └─────────────────────────────┘    │
       │                                                │
┌──────┴──┐    ┌─────────┐ ┌─────────┐ ┌─────────┐     │    ┌───────┐    ┌───────┐
│  User   │───▶│ Model A │ │ Model B │ │ Model C │─────┼───▶│  TUI  │───▶│ Judge │
│ Prompt  │    └─────────┘ └─────────┘ └─────────┘     │    └───────┘    └───────┘
└─────────┘                                             │
                    All calls are concurrent (tokio)    │
```

1. **Dispatch** — Your prompt is sent to all configured models simultaneously
2. **Collect** — Responses stream back via mpsc channels and render in the TUI
3. **Judge** — All responses are sent to the judge model for structured scoring
4. **Report** — Scoreboard with rankings and per-criteria reasoning

## Scoring Criteria

The judge evaluates each response on a 1-10 scale:

| Criterion | What it measures |
|-----------|-----------------|
| **Accuracy** | Is the information correct and factual? |
| **Helpfulness** | Does it address what the user actually needs? |
| **Clarity** | Is it well-structured and easy to understand? |
| **Creativity** | Does it show original thinking or novel approaches? |
| **Conciseness** | Is it appropriately detailed without being verbose? |

**Total: /50** — The judge provides brief reasoning for each score.

> **Tip:** Use a different judge model than the contestants to reduce self-preference bias. If you're comparing Claude vs GPT, use Gemini as the judge.

## Tips

- **Compare anything** — same family (gpt-4o vs gpt-4o-mini), cross-provider (Claude vs GPT vs Gemini), or local vs cloud (Llama vs GPT-4o)
- **Run the same prompt multiple times** — LLM outputs are non-deterministic, so scores will vary
- **Use `--json` for automation** — pipe to `jq`, build dashboards, track model quality over time
- **Per-model overrides** — point specific models at different endpoints (e.g., one model direct, rest through proxy)
- **No model limit** — compare 2 models or 20; the TUI grid auto-layouts

## Roadmap

- [x] ~~JSON output mode~~
- [x] ~~Per-model latency tracking~~
- [x] ~~Unified OpenAI-compatible endpoint (LiteLLM/OpenRouter)~~
- [x] ~~Dynamic model list (no code changes to add models)~~
- [x] ~~Auto-layout TUI grid for any number of models~~
- [ ] Multi-turn conversation support
- [ ] Token usage and cost tracking per model
- [ ] Configurable scoring criteria
- [ ] Export results to CSV
- [ ] Time-to-first-token latency tracking
- [ ] Side-by-side diff view for similar responses

## License

MIT

## Built With

- [Rust](https://www.rust-lang.org/) + [tokio](https://tokio.rs/) for async concurrency
- [ratatui](https://ratatui.rs/) + [crossterm](https://github.com/crossterm-rs/crossterm) for the terminal UI
- [reqwest](https://github.com/seanmonstar/reqwest) for HTTP
- [clap](https://github.com/clap-rs/clap) for CLI argument parsing
- [LiteLLM](https://github.com/BerriAI/litellm) / [OpenRouter](https://openrouter.ai) as the recommended proxy
