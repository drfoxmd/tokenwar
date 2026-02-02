use anyhow::{anyhow, Result};
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use dotenv::dotenv;
use futures_util::StreamExt;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    env,
    io::{self, Read},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(name = "tokenwar", version, about = "Compare LLM responses side-by-side")]
struct Args {
    /// Prompt text (if omitted, read from stdin)
    prompt: Option<String>,
    /// Stream responses as they arrive
    #[arg(long)]
    stream: bool,
    /// Request timeout in seconds
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,
    /// Disable the TUI and print plain output
    #[arg(long)]
    no_tui: bool,
    /// Output machine-readable JSON (conflicts with --no-tui)
    #[arg(long, conflicts_with = "no_tui")]
    json: bool,
}

#[derive(Clone, Debug)]
struct Config {
    anthropic_key: Option<String>,
    anthropic_model: Option<String>,
    openai_key: Option<String>,
    openai_model: Option<String>,
    grok_key: Option<String>,
    grok_model: Option<String>,
    gemini_key: Option<String>,
    gemini_model: Option<String>,
    generic_key: Option<String>,
    generic_model: Option<String>,
    generic_url: Option<String>,
    judge_provider: String,
    judge_model: String,
}

#[derive(Clone, Debug)]
enum Provider {
    Anthropic,
    OpenAI,
    Grok,
    Gemini,
    Generic,
}

impl Provider {
    fn name(&self) -> &'static str {
        match self {
            Provider::Anthropic => "Anthropic",
            Provider::OpenAI => "OpenAI",
            Provider::Grok => "Grok (xAI)",
            Provider::Gemini => "Gemini",
            Provider::Generic => "Generic",
        }
    }
}

#[derive(Error, Debug)]
enum ProviderError {
    #[error("missing configuration: {0}")]
    MissingConfig(&'static str),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Debug)]
struct ProviderUpdate {
    index: usize,
    append: Option<String>,
    done: bool,
    error: Option<String>,
    latency: Option<Duration>,
}

#[derive(Clone, Debug)]
struct ProviderResult {
    provider: Provider,
    text: String,
    error: Option<String>,
    latency: Option<Duration>,
}

#[derive(Serialize)]
struct JudgeRequest<'a> {
    prompt: &'a str,
    responses: Vec<JudgeResponse<'a>>,
}

#[derive(Serialize)]
struct JudgeResponse<'a> {
    provider: &'a str,
    response: &'a str,
}

#[derive(Deserialize, Debug, Default, Serialize)]
#[serde(default)]
struct JudgeScores {
    scores: Vec<JudgeScore>,
}

#[derive(Deserialize, Debug, Default, Serialize)]
#[serde(default)]
struct JudgeScore {
    provider: String,
    accuracy: ScoreItem,
    helpfulness: ScoreItem,
    clarity: ScoreItem,
    creativity: ScoreItem,
    conciseness: ScoreItem,
    #[serde(alias = "overall")]
    _overall: Option<ScoreItem>,
}

#[derive(Deserialize, Debug, Default, Serialize)]
#[serde(default)]
struct ScoreItem {
    score: f64,
    reasoning: String,
}

const JUDGE_SCHEMA_EXAMPLE: &str = r#"{
  "scores": [
    {
      "provider": "OpenAI",
      "accuracy": { "score": 8.5, "reasoning": "Mostly correct." },
      "helpfulness": { "score": 8.0, "reasoning": "Addresses the request." },
      "clarity": { "score": 7.5, "reasoning": "Readable and structured." },
      "creativity": { "score": 6.0, "reasoning": "Some novel framing." },
      "conciseness": { "score": 7.0, "reasoning": "Not too verbose." },
      "_overall": { "score": 7.4, "reasoning": "Solid overall." }
    }
  ]
}"#;

const JUDGE_PROMPT: &str = "You are an expert AI response evaluator. Given a user prompt and multiple AI responses, score each response on these criteria (1-10):\n- Accuracy: Is the information correct and factual?\n- Helpfulness: Does it address what the user actually needs?\n- Clarity: Is it well-structured and easy to understand?\n- Creativity: Does it show original thinking or novel approaches?\n- Conciseness: Is it appropriately detailed without being verbose?\n\nReturn ONLY valid JSON with the exact schema below. Do not wrap in markdown fences. Do not include extra text.\nSchema:\n{\n  \"scores\": [\n    {\n      \"provider\": string,\n      \"accuracy\": { \"score\": number, \"reasoning\": string },\n      \"helpfulness\": { \"score\": number, \"reasoning\": string },\n      \"clarity\": { \"score\": number, \"reasoning\": string },\n      \"creativity\": { \"score\": number, \"reasoning\": string },\n      \"conciseness\": { \"score\": number, \"reasoning\": string },\n      \"_overall\": { \"score\": number, \"reasoning\": string }  // optional\n    }\n  ]\n}\nExample:\n";

#[derive(Serialize)]
struct JsonOutput {
    prompt: String,
    providers: Vec<JsonProvider>,
    scores: Vec<JudgeScore>,
    metadata: JsonMetadata,
}

#[derive(Serialize)]
struct JsonProvider {
    name: String,
    model: Option<String>,
    response_text: String,
    error: Option<String>,
    latency_ms: Option<u64>,
}

#[derive(Serialize)]
struct JsonMetadata {
    timestamp: u64,
    timeout_secs: u64,
    stream: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let args = Args::parse();

    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                return Err(anyhow!("prompt is empty"));
            }
            trimmed.to_string()
        }
    };

    let config = load_config()?;
    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout_secs))
        .build()?;

    let providers = configured_providers(&config);
    if providers.len() < 2 {
        return Err(anyhow!(
            "need at least 2 configured providers; set at least two API keys and models (e.g. ANTHROPIC_API_KEY + ANTHROPIC_MODEL)"
        ));
    }

    let (tx, rx) = mpsc::channel::<ProviderUpdate>(128);
    let mut handles = Vec::new();

    for (index, provider) in providers.iter().cloned().enumerate() {
        let tx = tx.clone();
        let client = client.clone();
        let config = config.clone();
        let prompt = prompt.clone();
        let stream = args.stream;
        let handle = tokio::spawn(async move {
            let start = Instant::now();
            let result = match provider {
                Provider::Anthropic => call_anthropic(&client, &config, &prompt, stream, index, tx.clone()).await,
                Provider::OpenAI => call_openai(&client, &config, &prompt, stream, index, tx.clone()).await,
                Provider::Grok => call_grok(&client, &config, &prompt, stream, index, tx.clone()).await,
                Provider::Gemini => call_gemini(&client, &config, &prompt, stream, index, tx.clone()).await,
                Provider::Generic => call_generic(&client, &config, &prompt, stream, index, tx.clone()).await,
            };
            if let Err(err) = result {
                let _ = tx
                    .send(ProviderUpdate {
                        index,
                        append: None,
                        done: true,
                        error: Some(err.to_string()),
                        latency: Some(start.elapsed()),
                    })
                    .await;
            }
        });
        handles.push(handle);
    }
    drop(tx);

    let mut results = providers
        .iter()
        .cloned()
        .map(|provider| ProviderResult {
            provider,
            text: String::new(),
            error: None,
            latency: None,
        })
        .collect::<Vec<_>>();

    if args.json {
        collect_plain(rx, &mut results, false).await?;
    } else if args.no_tui {
        collect_plain(rx, &mut results, true).await?;
    } else {
        run_tui(rx, &mut results).await?;
    }

    for handle in handles {
        let _ = handle.await;
    }

    let judge = run_judge(&client, &config, &prompt, &results).await?;
    if args.json {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let providers = results
            .iter()
            .map(|result| JsonProvider {
                name: result.provider.name().to_string(),
                model: match result.provider {
                    Provider::Anthropic => config.anthropic_model.clone(),
                    Provider::OpenAI => config.openai_model.clone(),
                    Provider::Grok => config.grok_model.clone(),
                    Provider::Gemini => config.gemini_model.clone(),
                    Provider::Generic => config.generic_model.clone(),
                },
                response_text: result.text.trim().to_string(),
                error: result.error.clone(),
                latency_ms: result.latency.map(|latency| latency.as_millis() as u64),
            })
            .collect();
        let output = JsonOutput {
            prompt: prompt.clone(),
            providers,
            scores: judge.scores,
            metadata: JsonMetadata {
                timestamp,
                timeout_secs: args.timeout_secs,
                stream: args.stream,
            },
        };
        let json = serde_json::to_string_pretty(&output)?;
        println!("{}", json);
    } else {
        render_scoreboard(&judge);
    }

    Ok(())
}

fn load_config() -> Result<Config> {
    Ok(Config {
        anthropic_key: env::var("ANTHROPIC_API_KEY").ok(),
        anthropic_model: env::var("ANTHROPIC_MODEL").ok(),
        openai_key: env::var("OPENAI_API_KEY").ok(),
        openai_model: env::var("OPENAI_MODEL").ok(),
        grok_key: env::var("GROK_API_KEY").ok(),
        grok_model: env::var("GROK_MODEL").ok(),
        gemini_key: env::var("GEMINI_API_KEY").ok(),
        gemini_model: env::var("GEMINI_MODEL").ok(),
        generic_key: env::var("GENERIC_API_KEY").ok(),
        generic_model: env::var("GENERIC_MODEL").ok(),
        generic_url: env::var("GENERIC_API_URL").ok(),
        judge_provider: env::var("JUDGE_PROVIDER").unwrap_or_else(|_| "anthropic".to_string()),
        judge_model: env::var("JUDGE_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string()),
    })
}

fn configured_providers(config: &Config) -> Vec<Provider> {
    let mut providers = Vec::new();
    if config.anthropic_key.is_some() && config.anthropic_model.is_some() {
        providers.push(Provider::Anthropic);
    }
    if config.openai_key.is_some() && config.openai_model.is_some() {
        providers.push(Provider::OpenAI);
    }
    if config.grok_key.is_some() && config.grok_model.is_some() {
        providers.push(Provider::Grok);
    }
    if config.gemini_key.is_some() && config.gemini_model.is_some() {
        providers.push(Provider::Gemini);
    }
    if config.generic_key.is_some() && config.generic_model.is_some() && config.generic_url.is_some() {
        providers.push(Provider::Generic);
    }
    providers
}

async fn collect_plain(mut rx: mpsc::Receiver<ProviderUpdate>, results: &mut [ProviderResult], print_output: bool) -> Result<()> {
    while let Some(update) = rx.recv().await {
        let entry = &mut results[update.index];
        if let Some(chunk) = update.append {
            entry.text.push_str(&chunk);
        }
        if let Some(err) = update.error {
            entry.error = Some(err);
        }
        if let Some(latency) = update.latency {
            entry.latency = Some(latency);
        }
        if update.done {
            if print_output {
                let label = entry.provider.name();
                println!("\n=== {} ===", label);
                if let Some(err) = &entry.error {
                    println!("Error: {}", err);
                } else {
                    println!("{}", entry.text.trim());
                }
            }
        }
    }
    Ok(())
}

async fn run_tui(mut rx: mpsc::Receiver<ProviderUpdate>, results: &mut [ProviderResult]) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut done_count = 0usize;

    loop {
        while let Ok(update) = rx.try_recv() {
            let entry = &mut results[update.index];
            if let Some(chunk) = update.append {
                entry.text.push_str(&chunk);
            }
            if let Some(err) = update.error {
                entry.error = Some(err);
            }
            if let Some(latency) = update.latency {
                entry.latency = Some(latency);
            }
            if update.done {
                done_count += 1;
            }
        }

        terminal.draw(|f| {
            let size = f.size();
            match results.len() {
                0 => {}
                1 => render_panel(f, size, &results[0]),
                2 => {
                    let chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(size);
                    render_panel(f, chunks[0], &results[0]);
                    render_panel(f, chunks[1], &results[1]);
                }
                3 => {
                    let chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(33), Constraint::Percentage(34), Constraint::Percentage(33)])
                        .split(size);
                    render_panel(f, chunks[0], &results[0]);
                    render_panel(f, chunks[1], &results[1]);
                    render_panel(f, chunks[2], &results[2]);
                }
                4 => {
                    let rows = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(size);
                    let top = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(rows[0]);
                    let bottom = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(rows[1]);
                    render_panel(f, top[0], &results[0]);
                    render_panel(f, top[1], &results[1]);
                    render_panel(f, bottom[0], &results[2]);
                    render_panel(f, bottom[1], &results[3]);
                }
                _ => {
                    let rows = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
                        .split(size);
                    let top = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(33), Constraint::Percentage(34), Constraint::Percentage(33)])
                        .split(rows[0]);
                    let bottom = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(rows[1]);

                    render_panel(f, top[0], &results[0]);
                    render_panel(f, top[1], &results[1]);
                    render_panel(f, top[2], &results[2]);
                    render_panel(f, bottom[0], &results[3]);
                    render_panel(f, bottom[1], &results[4]);
                }
            }
        })?;

        if done_count >= results.len() {
            break;
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

fn render_panel(f: &mut ratatui::Frame, area: Rect, result: &ProviderResult) {
    let title = if let Some(_err) = &result.error {
        format!("{} (error)", result.provider.name())
    } else {
        result.provider.name().to_string()
    };
    let mut text = result.text.trim().to_string();
    if let Some(err) = &result.error {
        text = format!("Error: {}", err);
    }
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(Color::Cyan)))
        .borders(Borders::ALL);
    let paragraph = Paragraph::new(Text::from(text))
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

async fn call_anthropic(
    client: &Client,
    config: &Config,
    prompt: &str,
    stream: bool,
    index: usize,
    tx: mpsc::Sender<ProviderUpdate>,
) -> Result<()> {
    let start = Instant::now();
    let key = config
        .anthropic_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("ANTHROPIC_API_KEY"))?;
    let model = config
        .anthropic_model
        .as_ref()
        .ok_or(ProviderError::MissingConfig("ANTHROPIC_MODEL"))?;
    let body = serde_json::json!({
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    });
    let req = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body);

    if stream {
        let resp = req.send().await?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        continue;
                    }
                    if let Ok(value) = serde_json::from_str::<Value>(data) {
                        if value["type"] == "content_block_delta" {
                            if let Some(delta) = value["delta"]["text"].as_str() {
                                let _ = tx
                                    .send(ProviderUpdate {
                                        index,
                                        append: Some(delta.to_string()),
                                        done: false,
                                        error: None,
                                        latency: None,
                                    })
                                    .await;
                            }
                        }
                    }
                }
            }
        }
    } else {
        let resp = req.send().await?;
        let value: Value = resp.json().await?;
        let text = value["content"]
            .get(0)
            .and_then(|v| v["text"].as_str())
            .unwrap_or("")
            .to_string();
        let _ = tx
            .send(ProviderUpdate {
                index,
                append: Some(text),
                done: false,
                error: None,
                latency: None,
            })
            .await;
    }

    let _ = tx
        .send(ProviderUpdate {
            index,
            append: None,
            done: true,
            error: None,
            latency: Some(start.elapsed()),
        })
        .await;
    Ok(())
}

async fn call_openai(
    client: &Client,
    config: &Config,
    prompt: &str,
    stream: bool,
    index: usize,
    tx: mpsc::Sender<ProviderUpdate>,
) -> Result<()> {
    let key = config
        .openai_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("OPENAI_API_KEY"))?;
    let model = config
        .openai_model
        .as_ref()
        .ok_or(ProviderError::MissingConfig("OPENAI_MODEL"))?;
    call_openai_like(
        client,
        key,
        "https://api.openai.com/v1/chat/completions",
        model,
        prompt,
        stream,
        index,
        tx,
    )
    .await
}

async fn call_grok(
    client: &Client,
    config: &Config,
    prompt: &str,
    stream: bool,
    index: usize,
    tx: mpsc::Sender<ProviderUpdate>,
) -> Result<()> {
    let key = config
        .grok_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GROK_API_KEY"))?;
    let model = config
        .grok_model
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GROK_MODEL"))?;
    call_openai_like(
        client,
        key,
        "https://api.x.ai/v1/chat/completions",
        model,
        prompt,
        stream,
        index,
        tx,
    )
    .await
}

async fn call_generic(
    client: &Client,
    config: &Config,
    prompt: &str,
    stream: bool,
    index: usize,
    tx: mpsc::Sender<ProviderUpdate>,
) -> Result<()> {
    let key = config
        .generic_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GENERIC_API_KEY"))?;
    let model = config
        .generic_model
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GENERIC_MODEL"))?;
    let url = config
        .generic_url
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GENERIC_API_URL"))?;

    call_openai_like(client, key, url, model, prompt, stream, index, tx).await
}

async fn call_openai_like(
    client: &Client,
    api_key: &str,
    url: &str,
    model: &str,
    prompt: &str,
    stream: bool,
    index: usize,
    tx: mpsc::Sender<ProviderUpdate>,
) -> Result<()> {
    let start = Instant::now();
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": stream,
    });

    let req = client
        .post(url)
        .bearer_auth(api_key)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body);

    if stream {
        let resp = req.send().await?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        continue;
                    }
                    if let Ok(value) = serde_json::from_str::<Value>(data) {
                        if let Some(delta) = value["choices"][0]["delta"]["content"].as_str() {
                            let _ = tx
                                .send(ProviderUpdate {
                                    index,
                                    append: Some(delta.to_string()),
                                    done: false,
                                    error: None,
                                    latency: None,
                                })
                                .await;
                        }
                    }
                }
            }
        }
    } else {
        let resp = req.send().await?;
        let value: Value = resp.json().await?;
        let text = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let _ = tx
            .send(ProviderUpdate {
                index,
                append: Some(text),
                done: false,
                error: None,
                latency: None,
            })
            .await;
    }

    let _ = tx
        .send(ProviderUpdate {
            index,
            append: None,
            done: true,
            error: None,
            latency: Some(start.elapsed()),
        })
        .await;
    Ok(())
}

async fn call_gemini(
    client: &Client,
    config: &Config,
    prompt: &str,
    stream: bool,
    index: usize,
    tx: mpsc::Sender<ProviderUpdate>,
) -> Result<()> {
    let start = Instant::now();
    let key = config
        .gemini_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GEMINI_API_KEY"))?;
    let model = config
        .gemini_model
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GEMINI_MODEL"))?;
    let endpoint = if stream {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}",
            model, key
        )
    } else {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, key
        )
    };

    let body = serde_json::json!({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
    });

    let req = client
        .post(endpoint)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body);

    if stream {
        let resp = req.send().await?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(value) = serde_json::from_str::<Value>(line) {
                    if let Some(part) = value["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                        let _ = tx
                            .send(ProviderUpdate {
                                index,
                                append: Some(part.to_string()),
                                done: false,
                                error: None,
                                latency: None,
                            })
                            .await;
                    }
                }
            }
        }
    } else {
        let resp = req.send().await?;
        let value: Value = resp.json().await?;
        let text = value["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let _ = tx
            .send(ProviderUpdate {
                index,
                append: Some(text),
                done: false,
                error: None,
                latency: None,
            })
            .await;
    }

    let _ = tx
        .send(ProviderUpdate {
            index,
            append: None,
            done: true,
            error: None,
            latency: Some(start.elapsed()),
        })
        .await;
    Ok(())
}

async fn run_judge(client: &Client, config: &Config, prompt: &str, results: &[ProviderResult]) -> Result<JudgeScores> {
    let responses = results
        .iter()
        .map(|r| JudgeResponse {
            provider: r.provider.name(),
            response: if r.error.is_some() {
                "ERROR"
            } else {
                r.text.as_str()
            },
        })
        .collect::<Vec<_>>();

    let payload = JudgeRequest { prompt, responses };
    let judge_prompt = build_judge_prompt(prompt, &payload, false)?;
    let text = call_judge_with_prompt(client, config, &judge_prompt).await?;
    match parse_judge_scores(&text) {
        Ok(scores) => Ok(scores),
        Err(err) => {
            eprintln!("Judge JSON parse failed, retrying once: {}", err);
            let retry_prompt = build_judge_prompt(prompt, &payload, true)?;
            let retry_text = call_judge_with_prompt(client, config, &retry_prompt).await?;
            parse_judge_scores(&retry_text)
        }
    }
}

fn build_judge_prompt(prompt: &str, payload: &JudgeRequest<'_>, retry: bool) -> Result<String> {
    let retry_prefix = if retry {
        "Your previous response was invalid. Return ONLY valid JSON matching the schema below. No markdown, no prose.\n\n"
    } else {
        ""
    };
    Ok(format!(
        "{retry_prefix}{JUDGE_PROMPT}{example}\n\nUser prompt:\n{prompt}\n\nResponses:\n{payload}",
        example = JUDGE_SCHEMA_EXAMPLE,
        payload = serde_json::to_string_pretty(payload)?,
    ))
}

async fn call_judge_with_prompt(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    match config.judge_provider.as_str() {
        "anthropic" => call_anthropic_judge(client, config, prompt).await,
        "openai" => call_openai_judge(client, config, prompt).await,
        "grok" => call_grok_judge(client, config, prompt).await,
        "gemini" => call_gemini_judge(client, config, prompt).await,
        "generic" => call_generic_judge(client, config, prompt).await,
        other => Err(anyhow!("unsupported JUDGE_PROVIDER: {}", other)),
    }
}

fn parse_judge_scores(text: &str) -> Result<JudgeScores> {
    for candidate in extract_json_candidates(text) {
        if let Ok(scores) = serde_json::from_str::<JudgeScores>(&candidate) {
            return Ok(scores);
        }
    }
    Err(anyhow!("failed to parse judge JSON from response"))
}

fn extract_json_candidates(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    for candidate in extract_fenced_json(text) {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    for candidate in extract_braced_json(text) {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn extract_fenced_json(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let mut in_fence = false;
    let mut fence_lang: Option<String> = None;
    let mut buffer = String::new();

    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if in_fence {
                let lang = fence_lang.take().unwrap_or_default();
                let content = buffer.trim().to_string();
                if !content.is_empty()
                    && (lang.is_empty()
                        || lang.starts_with("json")
                        || content.contains('{'))
                {
                    candidates.push(content);
                }
                buffer.clear();
                in_fence = false;
            } else {
                let lang = trimmed.trim_start_matches("```").trim().to_lowercase();
                fence_lang = Some(lang);
                in_fence = true;
            }
            continue;
        }

        if in_fence {
            buffer.push_str(line);
            buffer.push('\n');
        }
    }

    if in_fence {
        let content = buffer.trim().to_string();
        if !content.is_empty() && content.contains('{') {
            candidates.push(content);
        }
    }

    candidates
}

fn extract_braced_json(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escape = false;

    for (idx, ch) in text.char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start_idx) = start.take() {
                            candidates.push(text[start_idx..=idx].to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    candidates
}

async fn call_anthropic_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let key = config
        .anthropic_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("ANTHROPIC_API_KEY"))?;
    let body = serde_json::json!({
        "model": config.judge_model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    });
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;
    let value: Value = resp.json().await?;
    Ok(value["content"][0]["text"].as_str().unwrap_or("").to_string())
}

async fn call_openai_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let key = config
        .openai_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("OPENAI_API_KEY"))?;
    let body = serde_json::json!({
        "model": config.judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    });
    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(key)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;
    let value: Value = resp.json().await?;
    Ok(value["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
}

async fn call_grok_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let key = config
        .grok_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GROK_API_KEY"))?;
    let body = serde_json::json!({
        "model": config.judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    });
    let resp = client
        .post("https://api.x.ai/v1/chat/completions")
        .bearer_auth(key)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;
    let value: Value = resp.json().await?;
    Ok(value["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
}

async fn call_gemini_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let key = config
        .gemini_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GEMINI_API_KEY"))?;
    let endpoint = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        config.judge_model, key
    );
    let body = serde_json::json!({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    });
    let resp = client
        .post(endpoint)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;
    let value: Value = resp.json().await?;
    Ok(value["candidates"][0]["content"]["parts"][0]["text"].as_str().unwrap_or("").to_string())
}

async fn call_generic_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let key = config
        .generic_key
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GENERIC_API_KEY"))?;
    let model = config
        .generic_model
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GENERIC_MODEL"))?;
    let url = config
        .generic_url
        .as_ref()
        .ok_or(ProviderError::MissingConfig("GENERIC_API_URL"))?;
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    });
    let resp = client
        .post(url)
        .bearer_auth(key)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;
    let value: Value = resp.json().await?;
    Ok(value["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
}

fn render_scoreboard(scores: &JudgeScores) {
    println!("\n=== Scoreboard ===");
    let mut totals: Vec<(String, f64)> = scores
        .scores
        .iter()
        .map(|s| {
            let total = s.accuracy.score
                + s.helpfulness.score
                + s.clarity.score
                + s.creativity.score
                + s.conciseness.score;
            (s.provider.clone(), total)
        })
        .collect();
    totals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (provider, total)) in totals.iter().enumerate() {
        println!("{}. {} - {:.1}/50", rank + 1, provider, total);
    }

    println!("\n=== Details ===");
    for score in &scores.scores {
        println!(
            "\n{}:\n  Accuracy: {:.1} ({})\n  Helpfulness: {:.1} ({})\n  Clarity: {:.1} ({})\n  Creativity: {:.1} ({})\n  Conciseness: {:.1} ({})",
            score.provider,
            score.accuracy.score,
            score.accuracy.reasoning,
            score.helpfulness.score,
            score.helpfulness.reasoning,
            score.clarity.score,
            score.clarity.reasoning,
            score.creativity.score,
            score.creativity.reasoning,
            score.conciseness.score,
            score.conciseness.reasoning
        );
    }
}
