# ⚔️ TokenWar

**Compare LLM responses side-by-side in your terminal, then let an AI judge score them.**

TokenWar sends the same prompt to multiple LLM models via an OpenAI-compatible endpoint, displays their responses in a split-pane TUI, and runs an LLM-as-judge evaluation scoring each response on accuracy, helpfulness, clarity, creativity, and conciseness.

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

**Example:** You're building a customer support bot. You write 10 representative prompts, run them through TokenWar, and discover that for *your specific domain*, gemini-2.5-flash outperforms gpt-4o while costing less. You'd never know this from public benchmarks.

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

- **⚡ Concurrent API calls** — All models queried simultaneously via tokio, not sequentially
- **📺 Terminal UI** — Split-pane ratatui display showing responses as they stream in
- **🏆 LLM-as-judge scoring** — Automated evaluation on 5 criteria (1-10 scale each, 50 max)
- **🔌 Dynamic model list** — Compare any set of models via a single OpenAI-compatible endpoint
- **📡 Streaming mode** — Watch responses arrive token-by-token with `--stream`
- **📋 Plain text mode** — `--no-tui` for piping output or CI/automation
- **📊 JSON output** — `--json` for machine-readable results with latency data
- **⏱️ Latency tracking** — Per-model response time in milliseconds
- **⏱️ Configurable timeout** — `--timeout-secs` to control how long to wait
- **🔧 Fully configurable** — Models and judge model set via `.env`
- **💪 Fault tolerant** — One model failing doesn't kill the others

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- API key for an OpenAI-compatible endpoint and at least 2 models to compare

### Build

```bash
git clone https://github.com/drfoxmd/tokenwar.git
cd tokenwar
cargo build --release
```

The binary will be at `target/release/tokenwar`.

### Configuration

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Base URL for the OpenAI-compatible proxy (LiteLLM, OpenRouter, etc.)
BASE_URL=http://localhost:4000/v1
API_KEY=sk-litellm-xxx

# Comma-separated list of model names to compare (min 2)
MODELS=claude-sonnet-4-20250514,gpt-4o,grok-3,gemini-2.5-flash

# Judge config (still separate)
JUDGE_MODEL=claude-sonnet-4-20250514

# Optional per-model overrides (if some models need a different endpoint/key)
# MODEL_0_BASE_URL=https://api.openai.com/v1
# MODEL_0_API_KEY=sk-xxx
# MODEL_0_NAME=Claude Sonnet 4
```

> **Tip:** Point `BASE_URL` at LiteLLM, OpenRouter, Ollama, or any OpenAI-compatible proxy.

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
# Stream responses as they arrive (token-by-token in TUI)
tokenwar --stream "What is quantum computing?"

# Plain text output (no TUI, good for scripts/CI)
tokenwar --no-tui "Compare REST vs GraphQL"

# JSON output (machine-readable, includes latency)
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
      "name": "claude-sonnet-4-20250514",
      "model": "claude-sonnet-4-20250514",
      "response_text": "2 + 2 = 4.",
      "error": null,
      "latency_ms": 1234
    },
    {
      "name": "gpt-4o",
      "model": "gpt-4o",
      "response_text": "The answer is 4.",
      "error": null,
      "latency_ms": 987
    }
  ],
  "scores": {
    "scores": []
  },
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

### Example: Plain Text Output

```
$ tokenwar --no-tui "What is the capital of France?"

=== claude-sonnet-4 ===
The capital of France is Paris...

=== gpt-4o ===
Paris is the capital of France...

=== grok-3 ===
The capital of France is Paris...

=== gemini-2.5-flash ===
Paris is the capital city of France...

=== llama-3.1-70b ===
The capital of France is Paris...

=== Scoreboard ===
1. claude-sonnet-4 - 43.0/50
2. gemini-2.5-flash - 42.0/50
3. gpt-4o - 41.5/50
4. grok-3 - 40.0/50
5. llama-3.1-70b - 39.5/50

=== Details ===
...
```

## Architecture

```
                    ┌─────────────────────────────┐
          prompt    │ OpenAI-compatible endpoint │
       ┌───────────▶│ (LiteLLM, OpenRouter, etc.)│────┐
       │            └─────────────────────────────┘    │
┌──────┴──┐         ┌──────────┐  ┌──────────┐         │     ┌───────────┐     ┌────────────┐
│  User   │────────▶│ Model A  │  │ Model B  │─────────┼────▶│    TUI    │────▶│   Judge    │
│ Prompt  │         └──────────┘  └──────────┘         │     │  Display  │     │  Scoring   │
└─────────┘         ┌──────────┐  ┌──────────┐         │     └───────────┘     └────────────┘
       │───────────▶│ Model C  │  │ Model D  │─────────┘
                    └──────────┘  └──────────┘

                    All calls are concurrent (tokio async)
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

**Total: /50** — The judge also provides brief reasoning for each score.

> **Note:** The judge itself is an LLM, so scores have inherent subjectivity. For best results, use a strong model (Claude Sonnet, GPT-4o) as the judge, and ideally a different model than the contestants to reduce self-preference bias.

## Tips

- **Use a different judge model** than the contestants to reduce self-preference bias
- **Run the same prompt multiple times** — LLM outputs are non-deterministic, so scores will vary
- **Point `BASE_URL` at any OpenAI-compatible proxy** — LiteLLM, OpenRouter, Ollama, or your own endpoint
- **Use `--json` for automation** — pipe output to `jq`, parse scores programmatically, build dashboards
- **Use `--no-tui` for simple text output** — pipe to files or grep through results
- **Ensure 2+ models in `MODELS`** — TokenWar validates this on startup

## Roadmap

- [x] ~~JSON output mode for programmatic consumption~~
- [x] ~~Model latency comparison (total response time)~~
- [x] ~~Gracefully skip models with missing API keys~~
- [ ] Multi-turn conversation support
- [ ] Token usage and cost tracking per model
- [ ] Configurable scoring criteria
- [ ] Export results to CSV
- [ ] Time-to-first-token latency tracking

## License

MIT

## Built With

- [Rust](https://www.rust-lang.org/) + [tokio](https://tokio.rs/) for async concurrency
- [ratatui](https://ratatui.rs/) + [crossterm](https://github.com/crossterm-rs/crossterm) for the terminal UI
- [reqwest](https://github.com/seanmonstar/reqwest) for HTTP
- [clap](https://github.com/clap-rs/clap) for CLI argument parsing
- Built by [Codex](https://openai.com/codex) 🤖, orchestrated by [Fox](https://github.com/clawdbot/clawdbot) 🦊
