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
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Span, Text},
    widgets::{Block, Borders, Paragraph, Widget, Wrap},
    Terminal,
};
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    env,
    io::{self, Read, Write},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(
    name = "tokenwar",
    version,
    about = "Compare LLM responses side-by-side"
)]
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
    /// Stream newline-delimited JSON events (conflicts with --json, --no-tui, --screenshot)
    #[arg(long, conflicts_with_all = ["json", "no_tui", "screenshot"])]
    json_stream: bool,
    /// Save a PNG screenshot of the final TUI state (optional path)
    #[arg(
        long,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = "tokenwar_screenshot.png",
        conflicts_with_all = ["no_tui", "json", "json_stream"]
    )]
    screenshot: Option<String>,
}

#[derive(Clone, Debug)]
struct ModelConfig {
    name: String,
    model: String,
    base_url: String,
    api_key: String,
}

#[derive(Clone, Debug)]
struct Config {
    models: Vec<ModelConfig>,
    judge_model: String,
    judge_base_url: String,
    judge_api_key: String,
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
    model: ModelConfig,
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
      "provider": "gpt-4o",
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

    let (tx, rx) = mpsc::channel::<ProviderUpdate>(128);
    let mut handles = Vec::new();

    for (index, model) in config.models.iter().cloned().enumerate() {
        let tx = tx.clone();
        let client = client.clone();
        let prompt = prompt.clone();
        let stream = args.stream;
        let handle = tokio::spawn(async move {
            let start = Instant::now();
            let result = call_model(&client, &model, &prompt, stream, index, tx.clone()).await;
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

    let mut results = config
        .models
        .iter()
        .cloned()
        .map(|model| ProviderResult {
            model,
            text: String::new(),
            error: None,
            latency: None,
        })
        .collect::<Vec<_>>();

    let ran_tui = !args.json && !args.no_tui && !args.json_stream;
    if args.json_stream {
        let start_event = serde_json::json!({
            "event": "start",
            "prompt": prompt.clone(),
            "models": config.models.iter().map(|model| model.model.clone()).collect::<Vec<_>>(),
        });
        emit_event(&start_event);
        collect_json_stream(rx, &mut results).await?;
    } else if args.json {
        collect_plain(rx, &mut results, false).await?;
    } else if args.no_tui {
        collect_plain(rx, &mut results, true).await?;
    } else {
        run_tui(rx, &mut results, args.screenshot.clone()).await?;
    }

    if ran_tui {
        for handle in &handles {
            handle.abort();
        }
    }
    for handle in handles {
        let _ = handle.await;
    }

    if ran_tui {
        for result in results.iter_mut() {
            if result.text.trim().is_empty() && result.error.is_none() {
                result.error = Some("Generation cancelled by user".to_string());
            }
        }
        let all_cancelled = results.iter().all(|result| result.text.trim().is_empty());
        if all_cancelled {
            return Ok(());
        }
    }

    if args.json_stream {
        let judging_event = serde_json::json!({ "event": "judging" });
        emit_event(&judging_event);
    }

    let judge = run_judge(&client, &config, &prompt, &results).await?;
    if args.json_stream {
        let scores_event = serde_json::json!({
            "event": "scores",
            "scores": judge.scores,
        });
        emit_event(&scores_event);
        let done_event = serde_json::json!({ "event": "done" });
        emit_event(&done_event);
    } else if args.json {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let providers = results
            .iter()
            .map(|result| JsonProvider {
                name: result.model.name.clone(),
                model: Some(result.model.model.clone()),
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

fn emit_event(value: &serde_json::Value) {
    let json = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    println!("{}", json);
    io::stdout().flush().unwrap();
}

fn load_config() -> Result<Config> {
    let base_url = env::var("BASE_URL")
        .map_err(|_| anyhow!("BASE_URL is required (e.g. http://localhost:4000/v1)"))?;
    let api_key = env::var("API_KEY").map_err(|_| anyhow!("API_KEY is required"))?;
    if base_url.trim().is_empty() {
        return Err(anyhow!("BASE_URL is empty"));
    }
    if api_key.trim().is_empty() {
        return Err(anyhow!("API_KEY is empty"));
    }
    let models_raw = env::var("MODELS")
        .map_err(|_| anyhow!("MODELS is required (comma-separated model list)"))?;

    let models_list = models_raw
        .split(',')
        .map(|entry| entry.trim().to_string())
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();

    if models_list.len() < 2 {
        return Err(anyhow!(
            "need at least 2 configured models; set MODELS with two or more entries"
        ));
    }

    let mut models = Vec::with_capacity(models_list.len());
    for (idx, model) in models_list.into_iter().enumerate() {
        let name_key = format!("MODEL_{}_NAME", idx);
        let base_key = format!("MODEL_{}_BASE_URL", idx);
        let api_key_var = format!("MODEL_{}_API_KEY", idx);

        let name = env::var(&name_key).unwrap_or_else(|_| model.clone());
        let model_base_url = env::var(&base_key).unwrap_or_else(|_| base_url.clone());
        let model_api_key = env::var(&api_key_var).unwrap_or_else(|_| api_key.clone());

        if model_base_url.trim().is_empty() {
            return Err(anyhow!("{} is empty", base_key));
        }
        if model_api_key.trim().is_empty() {
            return Err(anyhow!("{} is empty", api_key_var));
        }

        models.push(ModelConfig {
            name,
            model,
            base_url: model_base_url,
            api_key: model_api_key,
        });
    }

    let judge_model = env::var("JUDGE_MODEL").map_err(|_| anyhow!("JUDGE_MODEL is required"))?;
    let judge_base_url = env::var("JUDGE_BASE_URL").unwrap_or(base_url);
    let judge_api_key = env::var("JUDGE_API_KEY").unwrap_or(api_key);

    if judge_base_url.trim().is_empty() {
        return Err(anyhow!("JUDGE_BASE_URL is empty"));
    }
    if judge_api_key.trim().is_empty() {
        return Err(anyhow!("JUDGE_API_KEY is empty"));
    }

    Ok(Config {
        models,
        judge_model,
        judge_base_url,
        judge_api_key,
    })
}

async fn collect_plain(
    mut rx: mpsc::Receiver<ProviderUpdate>,
    results: &mut [ProviderResult],
    print_output: bool,
) -> Result<()> {
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
                let label = entry.model.name.as_str();
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

async fn collect_json_stream(
    mut rx: mpsc::Receiver<ProviderUpdate>,
    results: &mut [ProviderResult],
) -> Result<()> {
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
            let provider = entry.model.name.clone();
            let model = entry.model.model.clone();
            let response_text = entry.text.trim().to_string();
            let error = entry.error.clone();
            let latency_ms = entry.latency.map(|lat| lat.as_millis() as u64);
            let response_event = serde_json::json!({
                "event": "response",
                "provider": provider,
                "model": model,
                "response_text": response_text,
                "error": error,
                "latency_ms": latency_ms,
            });
            emit_event(&response_event);
        }
    }
    Ok(())
}

async fn run_tui(
    mut rx: mpsc::Receiver<ProviderUpdate>,
    results: &mut [ProviderResult],
    screenshot_path: Option<String>,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut focused_index = 0usize;
    let mut scroll_offsets = vec![0usize; results.len()];
    let mut fullscreen = false;
    let mut last_size = None;
    let start_time = Instant::now();
    let mut frame_counter: u64 = 0;

    let result = (|| -> Result<()> {
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
            }

            if !results.is_empty() {
                focused_index = focused_index.min(results.len().saturating_sub(1));
            }

            let size = terminal.size()?;
            last_size = Some(size);
            let panels = if fullscreen {
                vec![size]
            } else {
                layout_panels(size, results.len())
            };

            if !results.is_empty() {
                let line_count = |result: &ProviderResult| -> usize {
                    if result.error.is_some() {
                        return 1;
                    }
                    let trimmed = result.text.trim();
                    let count = trimmed.lines().count();
                    if count == 0 { 1 } else { count }
                };

                if fullscreen {
                    let result = &results[focused_index];
                    let total_lines = line_count(result);
                    let visible_lines = panels[0].height.saturating_sub(2) as usize;
                    let visible_lines = visible_lines.max(1);
                    let max_offset = total_lines.saturating_sub(visible_lines);
                    scroll_offsets[focused_index] = scroll_offsets[focused_index].min(max_offset);
                } else {
                    for (idx, result) in results.iter().enumerate() {
                        let total_lines = line_count(result);
                        let area = panels.get(idx).copied().unwrap_or(size);
                        let visible_lines = area.height.saturating_sub(2) as usize;
                        let visible_lines = visible_lines.max(1);
                        let max_offset = total_lines.saturating_sub(visible_lines);
                        scroll_offsets[idx] = scroll_offsets[idx].min(max_offset);
                    }
                }
            }

            terminal.draw(|f| {
                if results.is_empty() {
                    return;
                }

                if fullscreen {
                    let result = &results[focused_index];
                    render_panel(
                        f,
                        panels[0],
                        result,
                        true,
                        scroll_offsets[focused_index],
                        frame_counter,
                        start_time.elapsed(),
                    );
                } else {
                    for (idx, rect) in panels.iter().copied().enumerate() {
                        if let Some(result) = results.get(idx) {
                            let focused = idx == focused_index;
                            render_panel(
                                f,
                                rect,
                                result,
                                focused,
                                scroll_offsets[idx],
                                frame_counter,
                                start_time.elapsed(),
                            );
                        }
                    }
                }
            })?;

            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => {
                            break;
                        }
                        KeyCode::Tab => {
                            if !results.is_empty() {
                                focused_index = (focused_index + 1) % results.len();
                            }
                        }
                        KeyCode::Char(' ') => {
                            if !results.is_empty() {
                                fullscreen = !fullscreen;
                            }
                        }
                        KeyCode::Char('j') | KeyCode::Down => {
                            if !results.is_empty() {
                                let area = if fullscreen {
                                    panels[0]
                                } else {
                                    panels[focused_index]
                                };
                                let result = &results[focused_index];
                                let total_lines = if result.error.is_some() {
                                    1
                                } else {
                                    let trimmed = result.text.trim();
                                    let count = trimmed.lines().count();
                                    if count == 0 { 1 } else { count }
                                };
                                let visible_lines = area.height.saturating_sub(2) as usize;
                                let visible_lines = visible_lines.max(1);
                                let max_offset = total_lines.saturating_sub(visible_lines);
                                scroll_offsets[focused_index] =
                                    (scroll_offsets[focused_index] + 1).min(max_offset);
                            }
                        }
                        KeyCode::Char('k') | KeyCode::Up => {
                            if !results.is_empty() {
                                scroll_offsets[focused_index] =
                                    scroll_offsets[focused_index].saturating_sub(1);
                            }
                        }
                        _ => {}
                    }
                }
            }

            frame_counter = frame_counter.wrapping_add(1);
        }

        Ok(())
    })();

    if let (Some(path), Some(size)) = (screenshot_path.as_deref(), last_size) {
        save_tui_screenshot(
            path,
            size,
            results,
            fullscreen,
            focused_index,
            &scroll_offsets,
            frame_counter,
            start_time.elapsed(),
        )?;
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    result
}

fn layout_panels(area: Rect, count: usize) -> Vec<Rect> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![area];
    }
    if count == 2 {
        return Layout::default()
            .direction(Direction::Horizontal)
            .constraints(equal_constraints(2))
            .split(area)
            .iter()
            .copied()
            .collect();
    }

    let columns = (count as f64).sqrt().ceil() as usize;
    let rows = (count + columns - 1) / columns;
    let row_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(equal_constraints(rows))
        .split(area);

    let mut panels = Vec::with_capacity(count);
    for &row in row_chunks.iter() {
        let col_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(equal_constraints(columns))
            .split(row);
        for &col in col_chunks.iter() {
            if panels.len() >= count {
                break;
            }
            panels.push(col);
        }
        if panels.len() >= count {
            break;
        }
    }

    panels
}

fn equal_constraints(count: usize) -> Vec<Constraint> {
    let count = count.max(1);
    let base = 100u16 / count as u16;
    let remainder = 100u16 % count as u16;
    (0..count)
        .map(|idx| {
            let extra = if (idx as u16) < remainder { 1 } else { 0 };
            Constraint::Percentage(base + extra)
        })
        .collect()
}

fn panel_text(result: &ProviderResult, frame: u64, elapsed: Duration) -> String {
    if let Some(err) = &result.error {
        return format!("Error: {}", err);
    }
    let trimmed = result.text.trim();
    if trimmed.is_empty() {
        let phases = ["Generating.", "Generating..", "Generating..."];
        let phase = ((frame / 6) as usize) % phases.len();
        let mut text = phases[phase].to_string();
        if result.latency.is_none() {
            text.push_str(&format!(" ({:.1}s)", elapsed.as_secs_f32()));
        }
        return text;
    }
    trimmed.to_string()
}

fn render_panel(
    f: &mut ratatui::Frame,
    area: Rect,
    result: &ProviderResult,
    focused: bool,
    scroll_offset: usize,
    frame: u64,
    elapsed: Duration,
) {
    let title_base = if let Some(_err) = &result.error {
        format!("{} (error)", result.model.name)
    } else {
        result.model.name.clone()
    };
    let text = panel_text(result, frame, elapsed);
    let total_lines = {
        let count = text.lines().count();
        if count == 0 { 1 } else { count }
    };
    let visible_lines = area.height.saturating_sub(2) as usize;
    let visible_lines = visible_lines.max(1);
    let max_offset = total_lines.saturating_sub(visible_lines);
    let clamped_offset = scroll_offset.min(max_offset);
    let indicator = format!("[{}/{}]", clamped_offset.saturating_add(1), total_lines);
    let title = format!("{} {}", title_base, indicator);
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let block = Block::default()
        .title(Span::styled(title, border_style))
        .borders(Borders::ALL)
        .border_style(border_style);
    let paragraph = Paragraph::new(Text::from(text))
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((clamped_offset as u16, 0));
    f.render_widget(paragraph, area);
}

fn render_panel_to_buffer(
    buffer: &mut Buffer,
    area: Rect,
    result: &ProviderResult,
    focused: bool,
    scroll_offset: usize,
    frame: u64,
    elapsed: Duration,
) {
    let title_base = if let Some(_err) = &result.error {
        format!("{} (error)", result.model.name)
    } else {
        result.model.name.clone()
    };
    let text = panel_text(result, frame, elapsed);
    let total_lines = {
        let count = text.lines().count();
        if count == 0 { 1 } else { count }
    };
    let visible_lines = area.height.saturating_sub(2) as usize;
    let visible_lines = visible_lines.max(1);
    let max_offset = total_lines.saturating_sub(visible_lines);
    let clamped_offset = scroll_offset.min(max_offset);
    let indicator = format!("[{}/{}]", clamped_offset.saturating_add(1), total_lines);
    let title = format!("{} {}", title_base, indicator);
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let block = Block::default()
        .title(Span::styled(title, border_style))
        .borders(Borders::ALL)
        .border_style(border_style);
    let paragraph = Paragraph::new(Text::from(text))
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((clamped_offset as u16, 0));
    paragraph.render(area, buffer);
}

fn buffer_to_text(buffer: &Buffer) -> String {
    let area = *buffer.area();
    let mut out = String::new();
    for y in 0..area.height {
        for x in 0..area.width {
            let cell = buffer.get(area.x + x, area.y + y);
            let symbol = cell.symbol();
            if symbol.is_empty() {
                out.push(' ');
            } else {
                out.push_str(symbol);
            }
        }
        if y + 1 < area.height {
            out.push('\n');
        }
    }
    out
}

fn save_tui_screenshot(
    path: &str,
    size: Rect,
    results: &[ProviderResult],
    fullscreen: bool,
    focused_index: usize,
    scroll_offsets: &[usize],
    frame: u64,
    elapsed: Duration,
) -> Result<()> {
    if results.is_empty() || size.width == 0 || size.height == 0 {
        return Ok(());
    }

    let mut buffer = Buffer::empty(Rect::new(0, 0, size.width, size.height));
    let panels = if fullscreen {
        vec![size]
    } else {
        layout_panels(size, results.len())
    };

    if fullscreen {
        let idx = focused_index.min(results.len().saturating_sub(1));
        render_panel_to_buffer(
            &mut buffer,
            panels[0],
            &results[idx],
            true,
            *scroll_offsets.get(idx).unwrap_or(&0),
            frame,
            elapsed,
        );
    } else {
        for (idx, rect) in panels.iter().copied().enumerate() {
            if let Some(result) = results.get(idx) {
                render_panel_to_buffer(
                    &mut buffer,
                    rect,
                    result,
                    idx == focused_index,
                    *scroll_offsets.get(idx).unwrap_or(&0),
                    frame,
                    elapsed,
                );
            }
        }
    }

    let text = buffer_to_text(&buffer);
    let font = ansee::Font {
        name: None,
        size: 14.0,
        line_height: 1.1,
    };
    let image = ansee::draw_image(&text, font)?;
    image.save(path)?;
    Ok(())
}

async fn call_model(
    client: &Client,
    model: &ModelConfig,
    prompt: &str,
    stream: bool,
    index: usize,
    tx: mpsc::Sender<ProviderUpdate>,
) -> Result<()> {
    let start = Instant::now();
    let _ = call_openai_like(
        client,
        &model.base_url,
        &model.api_key,
        &model.model,
        prompt,
        0.7,
        stream,
        Some((index, tx.clone())),
    )
    .await?;
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

async fn call_openai_like(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    prompt: &str,
    temperature: f64,
    stream: bool,
    tx: Option<(usize, mpsc::Sender<ProviderUpdate>)>,
) -> Result<String> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": stream,
    });

    let url = chat_completions_url(base_url);
    let req = client
        .post(url)
        .bearer_auth(api_key)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&body);

    let mut collected = String::new();

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
                            collected.push_str(delta);
                            if let Some((index, sender)) = &tx {
                                let _ = sender
                                    .send(ProviderUpdate {
                                        index: *index,
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
        let text = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        collected.push_str(&text);
        if let Some((index, sender)) = &tx {
            let _ = sender
                .send(ProviderUpdate {
                    index: *index,
                    append: Some(text),
                    done: false,
                    error: None,
                    latency: None,
                })
                .await;
        }
    }

    Ok(collected)
}

fn chat_completions_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{}/chat/completions", trimmed)
    }
}

async fn run_judge(
    client: &Client,
    config: &Config,
    prompt: &str,
    results: &[ProviderResult],
) -> Result<JudgeScores> {
    let responses = results
        .iter()
        .map(|r| JudgeResponse {
            provider: r.model.name.as_str(),
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
    call_openai_like(
        client,
        &config.judge_base_url,
        &config.judge_api_key,
        &config.judge_model,
        prompt,
        0.2,
        false,
        None,
    )
    .await
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
                    && (lang.is_empty() || lang.starts_with("json") || content.contains('{'))
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
