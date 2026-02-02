use anyhow::{anyhow, Context, Result};
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
    time::{Duration, SystemTime, UNIX_EPOCH},
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
    anthropic_key: String,
    anthropic_model: String,
    openai_key: String,
    openai_model: String,
    grok_key: String,
    grok_model: String,
    gemini_key: String,
    gemini_model: String,
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
}

#[derive(Clone, Debug)]
struct ProviderResult {
    provider: Provider,
    text: String,
    error: Option<String>,
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

#[derive(Deserialize, Debug, Serialize)]
struct JudgeScores {
    scores: Vec<JudgeScore>,
}

#[derive(Deserialize, Debug, Serialize)]
struct JudgeScore {
    provider: String,
    accuracy: ScoreItem,
    helpfulness: ScoreItem,
    clarity: ScoreItem,
    creativity: ScoreItem,
    conciseness: ScoreItem,
    _overall: Option<ScoreItem>,
}

#[derive(Deserialize, Debug, Serialize)]
struct ScoreItem {
    score: f64,
    reasoning: String,
}

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
}

#[derive(Serialize)]
struct JsonMetadata {
    timestamp: u64,
    timeout_secs: u64,
    stream: bool,
}

const JUDGE_PROMPT: &str = "You are an expert AI response evaluator. Given a user prompt and multiple AI responses, score each response on these criteria (1-10):\n- Accuracy: Is the information correct and factual?\n- Helpfulness: Does it address what the user actually needs?\n- Clarity: Is it well-structured and easy to understand?\n- Creativity: Does it show original thinking or novel approaches?\n- Conciseness: Is it appropriately detailed without being verbose?\n\nProvide scores as JSON with brief reasoning for each.";

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

    let providers = vec![
        Provider::Anthropic,
        Provider::OpenAI,
        Provider::Grok,
        Provider::Gemini,
        Provider::Generic,
    ];

    let (tx, rx) = mpsc::channel::<ProviderUpdate>(128);
    let mut handles = Vec::new();

    for (index, provider) in providers.iter().cloned().enumerate() {
        let tx = tx.clone();
        let client = client.clone();
        let config = config.clone();
        let prompt = prompt.clone();
        let stream = args.stream;
        let handle = tokio::spawn(async move {
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
                    })
                    .await;
            }
        });
        handles.push(handle);
    }
    drop(tx);

    let mut results = vec![
        ProviderResult {
            provider: Provider::Anthropic,
            text: String::new(),
            error: None,
        },
        ProviderResult {
            provider: Provider::OpenAI,
            text: String::new(),
            error: None,
        },
        ProviderResult {
            provider: Provider::Grok,
            text: String::new(),
            error: None,
        },
        ProviderResult {
            provider: Provider::Gemini,
            text: String::new(),
            error: None,
        },
        ProviderResult {
            provider: Provider::Generic,
            text: String::new(),
            error: None,
        },
    ];

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
                    Provider::Anthropic => Some(config.anthropic_model.clone()),
                    Provider::OpenAI => Some(config.openai_model.clone()),
                    Provider::Grok => Some(config.grok_model.clone()),
                    Provider::Gemini => Some(config.gemini_model.clone()),
                    Provider::Generic => config.generic_model.clone(),
                },
                response_text: result.text.trim().to_string(),
                error: result.error.clone(),
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
        anthropic_key: env::var("ANTHROPIC_API_KEY").context("missing ANTHROPIC_API_KEY")?,
        anthropic_model: env::var("ANTHROPIC_MODEL").context("missing ANTHROPIC_MODEL")?,
        openai_key: env::var("OPENAI_API_KEY").context("missing OPENAI_API_KEY")?,
        openai_model: env::var("OPENAI_MODEL").context("missing OPENAI_MODEL")?,
        grok_key: env::var("GROK_API_KEY").context("missing GROK_API_KEY")?,
        grok_model: env::var("GROK_MODEL").context("missing GROK_MODEL")?,
        gemini_key: env::var("GEMINI_API_KEY").context("missing GEMINI_API_KEY")?,
        gemini_model: env::var("GEMINI_MODEL").context("missing GEMINI_MODEL")?,
        generic_key: env::var("GENERIC_API_KEY").ok(),
        generic_model: env::var("GENERIC_MODEL").ok(),
        generic_url: env::var("GENERIC_API_URL").ok(),
        judge_provider: env::var("JUDGE_PROVIDER").unwrap_or_else(|_| "anthropic".to_string()),
        judge_model: env::var("JUDGE_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string()),
    })
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
            if update.done {
                done_count += 1;
            }
        }

        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
                .split(size);
            let top = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(33), Constraint::Percentage(34), Constraint::Percentage(33)])
                .split(chunks[0]);
            let bottom = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[1]);

            render_panel(f, top[0], &results[0]);
            render_panel(f, top[1], &results[1]);
            render_panel(f, top[2], &results[2]);
            render_panel(f, bottom[0], &results[3]);
            render_panel(f, bottom[1], &results[4]);
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
    let body = serde_json::json!({
        "model": config.anthropic_model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    });
    let req = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &config.anthropic_key)
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
            })
            .await;
    }

    let _ = tx
        .send(ProviderUpdate {
            index,
            append: None,
            done: true,
            error: None,
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
    call_openai_like(
        client,
        &config.openai_key,
        "https://api.openai.com/v1/chat/completions",
        &config.openai_model,
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
    call_openai_like(
        client,
        &config.grok_key,
        "https://api.x.ai/v1/chat/completions",
        &config.grok_model,
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
            })
            .await;
    }

    let _ = tx
        .send(ProviderUpdate {
            index,
            append: None,
            done: true,
            error: None,
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
    let endpoint = if stream {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}",
            config.gemini_model, config.gemini_key
        )
    } else {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            config.gemini_model, config.gemini_key
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
            })
            .await;
    }

    let _ = tx
        .send(ProviderUpdate {
            index,
            append: None,
            done: true,
            error: None,
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
    let judge_prompt = format!(
        "{}\n\nUser prompt:\n{}\n\nResponses:\n{}",
        JUDGE_PROMPT,
        prompt,
        serde_json::to_string_pretty(&payload)?
    );

    match config.judge_provider.as_str() {
        "anthropic" => {
            let text = call_anthropic_judge(client, config, &judge_prompt).await?;
            parse_judge_scores(&text)
        }
        "openai" => {
            let text = call_openai_judge(client, config, &judge_prompt).await?;
            parse_judge_scores(&text)
        }
        "grok" => {
            let text = call_grok_judge(client, config, &judge_prompt).await?;
            parse_judge_scores(&text)
        }
        "gemini" => {
            let text = call_gemini_judge(client, config, &judge_prompt).await?;
            parse_judge_scores(&text)
        }
        "generic" => {
            let text = call_generic_judge(client, config, &judge_prompt).await?;
            parse_judge_scores(&text)
        }
        other => Err(anyhow!("unsupported JUDGE_PROVIDER: {}", other)),
    }
}

fn parse_judge_scores(text: &str) -> Result<JudgeScores> {
    let json_start = text.find('{').ok_or_else(|| anyhow!("judge returned no JSON"))?;
    let json_str = &text[json_start..];
    let scores: JudgeScores = serde_json::from_str(json_str).context("failed to parse judge JSON")?;
    Ok(scores)
}

async fn call_anthropic_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let body = serde_json::json!({
        "model": config.judge_model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    });
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &config.anthropic_key)
        .header("anthropic-version", "2023-06-01")
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;
    let value: Value = resp.json().await?;
    Ok(value["content"][0]["text"].as_str().unwrap_or("").to_string())
}

async fn call_openai_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let body = serde_json::json!({
        "model": config.judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    });
    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(&config.openai_key)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;
    let value: Value = resp.json().await?;
    Ok(value["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
}

async fn call_grok_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let body = serde_json::json!({
        "model": config.judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    });
    let resp = client
        .post("https://api.x.ai/v1/chat/completions")
        .bearer_auth(&config.grok_key)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body)
        .send()
        .await?;
    let value: Value = resp.json().await?;
    Ok(value["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
}

async fn call_gemini_judge(client: &Client, config: &Config, prompt: &str) -> Result<String> {
    let endpoint = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        config.judge_model, config.gemini_key
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
