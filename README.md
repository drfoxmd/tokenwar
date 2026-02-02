# ⚔️ TokenWar

**Compare LLM responses side-by-side in your terminal, then let an AI judge score them.**

TokenWar sends the same prompt to multiple LLM providers simultaneously, displays their responses in a split-pane TUI, and runs an LLM-as-judge evaluation scoring each response on accuracy, helpfulness, clarity, creativity, and conciseness.

```
┌─────────────────────┬──────────────────────┬─────────────────────┐
│ Anthropic           │ OpenAI               │ Grok (xAI)          │
│                     │                      │                     │
│ The Rust ownership  │ Rust's ownership     │ In Rust, ownership  │
│ system ensures      │ model is a set of    │ is the core concept │
│ memory safety       │ rules that the       │ that makes memory   │
│ without a garbage   │ compiler checks at   │ safe without GC...  │
│ collector...        │ compile time...      │                     │
│                     │                      │                     │
├─────────────────────┴───────────┬──────────┴─────────────────────┤
│ Gemini                          │ Generic                        │
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
1. Anthropic - 42.0/50
2. Gemini - 40.5/50
3. OpenAI - 39.0/50
4. Grok (xAI) - 38.5/50
5. Generic - 37.0/50

=== Details ===

Anthropic:
  Accuracy: 9.0 (Correct and precise explanation of ownership rules)
  Helpfulness: 8.5 (Directly addresses the question with practical examples)
  Clarity: 8.5 (Well-structured with clear progression of concepts)
  Creativity: 8.0 (Novel analogy comparing ownership to real-world lending)
  Conciseness: 8.0 (Thorough but not verbose)

OpenAI:
  Accuracy: 8.5 (Accurate coverage of core concepts)
  Helpfulness: 8.0 (Good overview but fewer practical examples)
  ...
```

## Why TokenWar?

### When it's better than just using Claude or ChatGPT

| Use Case | Why TokenWar Wins |
|----------|---------------|
| **Evaluating models for your use case** | See how 5 models handle *your* actual prompts, not benchmarks |
| **Reducing bias in model selection** | An independent judge scores responses — not your gut feeling |
| **Catching hallucinations** | If 4 models agree and 1 doesn't, you've found a hallucination |
| **Prompt engineering** | Instantly see how different models interpret the same prompt |
| **Choosing a provider for production** | Real response quality + latency data, not marketing claims |
| **Creative work** | Compare writing styles, get 5 different angles on the same topic |
| **Factual research** | Cross-reference answers across providers for higher confidence |
| **Cost optimization** | If a cheaper model scores comparably, you've found your winner |

**Example:** You're building a customer support bot. You write 10 representative prompts, run them through TokenWar, and discover that for *your specific domain*, Gemini outperforms GPT-4o while costing less. You'd never know this from public benchmarks.

### When you should just use Claude or ChatGPT

| Situation | Why TokenWar is Overkill |
|-----------|----------------------|
| **Quick one-off questions** | You just need an answer, not a comparison |
| **Conversational/multi-turn chat** | TokenWar is single-turn only — no follow-ups |
| **You already know your preferred model** | No need to compare if you're happy |
| **Cost-sensitive usage** | TokenWar calls 5 APIs + a judge = 6x the cost of one model |
| **Image/audio/video tasks** | TokenWar is text-only |
| **You need tool use or function calling** | TokenWar sends plain prompts, no tool schemas |

## Features

- **⚡ Concurrent API calls** — All providers queried simultaneously via tokio, not sequentially
- **📺 Terminal UI** — Split-pane ratatui display showing responses as they stream in
- **🏆 LLM-as-judge scoring** — Automated evaluation on 5 criteria (1-10 scale each, 50 max)
- **🔌 5 providers** — Anthropic, OpenAI, Grok/xAI, Google Gemini, and any OpenAI-compatible API
- **📡 Streaming mode** — Watch responses arrive token-by-token with `--stream`
- **📋 Plain text mode** — `--no-tui` for piping output or CI/automation
- **📊 JSON output** — `--json` for machine-readable results with latency data
- **⏱️ Latency tracking** — Per-provider response time in milliseconds
- **⏱️ Configurable timeout** — `--timeout-secs` to control how long to wait
- **🔧 Fully configurable** — Models, judge provider, and judge model all set via `.env`
- **💪 Fault tolerant** — One provider failing doesn't kill the others

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- API keys for at least 2 providers (the more the merrier)

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
# Required: at least set the providers you want to use
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
GROK_API_KEY=xai-...
GROK_MODEL=grok-3
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash

# Optional: any OpenAI-compatible API (e.g., Ollama, Together, Groq)
GENERIC_API_KEY=sk-...
GENERIC_MODEL=llama-3.1-70b
GENERIC_API_URL=https://api.together.xyz/v1/chat/completions

# Judge configuration (which provider/model evaluates the responses)
JUDGE_PROVIDER=anthropic
JUDGE_MODEL=claude-sonnet-4-20250514
```

> **Tip:** The Generic slot accepts any OpenAI-compatible API. Use it for Ollama (`http://localhost:11434/v1/chat/completions`), Together AI, Groq, Fireworks, or your own endpoint.

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
  "timestamp": 1738492800,
  "providers": [
    {
      "name": "Anthropic",
      "model": "claude-sonnet-4-20250514",
      "response_text": "2 + 2 = 4.",
      "error": null,
      "latency_ms": 1234
    },
    {
      "name": "OpenAI",
      "model": "gpt-4o",
      "response_text": "The answer is 4.",
      "error": null,
      "latency_ms": 987
    }
  ],
  "judge_scores": {
    "scores": [...]
  },
  "settings": {
    "timeout_secs": 60,
    "stream": false
  }
}
```

### TUI Controls

| Key | Action |
|-----|--------|
| `q` | Quit early (skips waiting for remaining responses) |

The TUI automatically exits once all providers have responded, then displays the judge scoreboard.

### Example: Plain Text Output

```
$ tokenwar --no-tui "What is the capital of France?"

=== Anthropic ===
The capital of France is Paris...

=== OpenAI ===
Paris is the capital of France...

=== Grok (xAI) ===
The capital of France is Paris...

=== Gemini ===
Paris is the capital city of France...

=== Generic ===
The capital of France is Paris...

=== Scoreboard ===
1. Anthropic - 43.0/50
2. Gemini - 42.0/50
3. OpenAI - 41.5/50
4. Grok (xAI) - 40.0/50
5. Generic - 39.5/50

=== Details ===
...
```

## Architecture

```
                    ┌──────────┐
          prompt    │          │
       ┌───────────▶│ Anthropic │──────┐
       │            │          │      │
       │            └──────────┘      │
       │            ┌──────────┐      │
       │───────────▶│  OpenAI  │──────│
       │            └──────────┘      │
┌──────┴──┐         ┌──────────┐      │     ┌───────────┐     ┌────────────┐
│  User   │────────▶│   Grok   │──────┼────▶│    TUI    │────▶│   Judge    │
│ Prompt  │         └──────────┘      │     │  Display  │     │  Scoring   │
└─────────┘         ┌──────────┐      │     └───────────┘     └────────────┘
       │───────────▶│  Gemini  │──────│
       │            └──────────┘      │
       │            ┌──────────┐      │
       └───────────▶│ Generic  │──────┘
                    └──────────┘

                    All calls are concurrent (tokio async)
```

1. **Dispatch** — Your prompt is sent to all configured providers simultaneously
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

> **Note:** The judge itself is an LLM, so scores have inherent subjectivity. For best results, use a strong model (Claude Sonnet, GPT-4o) as the judge, and ideally a different provider than the contestants to reduce self-preference bias.

## Tips

- **Use a different judge provider** than the contestants to avoid self-preference bias (e.g., if comparing Anthropic vs OpenAI models, use Gemini as the judge)
- **Run the same prompt multiple times** — LLM outputs are non-deterministic, so scores will vary
- **The Generic slot is versatile** — point it at Ollama for local models, or any OpenAI-compatible API
- **Use `--json` for automation** — pipe output to `jq`, parse scores programmatically, build dashboards
- **Use `--no-tui` for simple text output** — pipe to files or grep through results
- **Providers with missing keys are skipped** — you only need 2+ configured providers

## Roadmap

- [x] ~~JSON output mode for programmatic consumption~~
- [x] ~~Provider latency comparison (total response time)~~
- [x] ~~Gracefully skip providers with missing API keys~~
- [ ] Multi-turn conversation support
- [ ] Token usage and cost tracking per provider
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
