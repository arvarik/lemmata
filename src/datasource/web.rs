use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::process::Command;

use super::traits::{DataItem, DataSource};

/// Structured representation of a scraped web document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedDocument {
    pub url: String,
    pub title: String,
    pub markdown: String,
    pub word_count: usize,
    pub is_likely_spa: bool,
}

/// Options controlling web page fetching and extraction.
#[derive(Debug, Clone)]
pub struct ScrapeOptions {
    pub user_agent: String,
    pub timeout_secs: u64,
    pub max_chars: usize,
    pub use_jina: bool,
    pub jina_api_key: Option<String>,
}

impl Default for ScrapeOptions {
    fn default() -> Self {
        Self {
            user_agent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 (compatible; LemmaBot/0.1; +https://github.com/arvarik/lemma)".to_string(),
            timeout_secs: 15,
            max_chars: 4000,
            use_jina: false,
            jina_api_key: std::env::var("JINA_API_KEY").ok(),
        }
    }
}

/// Zero-dependency streaming HTML to Markdown state machine extractor.
pub struct HtmlExtractor;

impl HtmlExtractor {
    /// Extract clean, LLM-ready text and title from raw HTML.
    pub fn extract(html: &str, url: &str) -> ExtractedDocument {
        let title = Self::extract_title(html);
        let is_likely_spa = Self::detect_spa(html);
        let markdown = Self::html_to_markdown(html);
        let word_count = markdown.split_whitespace().count();

        ExtractedDocument {
            url: url.to_string(),
            title,
            markdown,
            word_count,
            is_likely_spa,
        }
    }

    /// Extract document title via OpenGraph, Twitter card, <title>, or <h1>.
    pub fn extract_title(html: &str) -> String {
        // 1. Try og:title
        if let Some(og) = Self::extract_meta_content(html, "property=\"og:title\"") {
            return Self::clean_title(&og);
        }
        if let Some(og) = Self::extract_meta_content(html, "property='og:title'") {
            return Self::clean_title(&og);
        }

        // 2. Try twitter:title
        if let Some(tw) = Self::extract_meta_content(html, "name=\"twitter:title\"") {
            return Self::clean_title(&tw);
        }
        if let Some(tw) = Self::extract_meta_content(html, "name='twitter:title'") {
            return Self::clean_title(&tw);
        }

        // 3. Try <title>...</title>
        if let Some(start) = Self::find_tag_case_insensitive(html, "<title") {
            let slice = &html[start..];
            if let Some(tag_end) = slice.find('>') {
                let content_start = tag_end + 1;
                let remainder = &slice[content_start..];
                if let Some(end) = Self::find_tag_case_insensitive(remainder, "</title>") {
                    let title = &remainder[..end];
                    return Self::clean_title(&Self::decode_entities(title.trim()));
                }
            }
        }

        // 4. Try <h1>...</h1>
        if let Some(start) = Self::find_tag_case_insensitive(html, "<h1") {
            let slice = &html[start..];
            if let Some(tag_end) = slice.find('>') {
                let content_start = tag_end + 1;
                let remainder = &slice[content_start..];
                if let Some(end) = Self::find_tag_case_insensitive(remainder, "</h1>") {
                    let h1 = &remainder[..end];
                    let clean = Self::html_to_markdown(h1);
                    if !clean.is_empty() {
                        return Self::clean_title(&clean);
                    }
                }
            }
        }

        String::new()
    }

    fn find_tag_case_insensitive(html: &str, tag_prefix: &str) -> Option<usize> {
        let tag_bytes = tag_prefix.as_bytes();
        let html_bytes = html.as_bytes();
        if html_bytes.len() < tag_bytes.len() {
            return None;
        }
        for i in 0..=(html_bytes.len() - tag_bytes.len()) {
            if html.is_char_boundary(i)
                && html_bytes[i..i + tag_bytes.len()]
                    .iter()
                    .zip(tag_bytes.iter())
                    .all(|(a, b)| a.eq_ignore_ascii_case(b))
            {
                return Some(i);
            }
        }
        None
    }

    fn extract_meta_content(html: &str, marker: &str) -> Option<String> {
        let pos = html.find(marker)?;
        let tag_start = html[..pos].rfind("<meta")?;
        let tag_end = html[pos..].find('>')? + pos;
        let tag_slice = &html[tag_start..tag_end];

        let content_marker = "content=\"";
        if let Some(c_pos) = tag_slice.find(content_marker) {
            let val_start = c_pos + content_marker.len();
            let val_end = tag_slice[val_start..].find('"')? + val_start;
            return Some(Self::decode_entities(&tag_slice[val_start..val_end]));
        }

        let content_marker_sq = "content='";
        if let Some(c_pos) = tag_slice.find(content_marker_sq) {
            let val_start = c_pos + content_marker_sq.len();
            let val_end = tag_slice[val_start..].find('\'')? + val_start;
            return Some(Self::decode_entities(&tag_slice[val_start..val_end]));
        }

        None
    }

    fn clean_title(title: &str) -> String {
        for sep in [" | ", " - ", " — ", " · ", " » ", " :: "] {
            if let Some(idx) = title.rfind(sep)
                && idx > 4
            {
                return title[..idx].trim().to_string();
            }
        }
        title.trim().to_string()
    }

    /// Detect Single-Page Application (SPA) shells requiring headless rendering.
    pub fn detect_spa(html: &str) -> bool {
        let spa_markers = [
            "<div id=\"root\"></div>",
            "<div id=\"app\"></div>",
            "<div id=\"__next\"></div>",
            "window.__INITIAL_STATE__",
            "Please enable JavaScript to view",
            "You need to enable JavaScript to run this app",
        ];
        spa_markers.iter().any(|m| html.contains(m))
    }

    /// Streaming parser turning HTML into structured Markdown without external crates.
    pub fn html_to_markdown(html: &str) -> String {
        let mut out = String::with_capacity(html.len() / 2);
        let mut chars = html.chars().peekable();

        let discard_tags: HashSet<&str> = [
            "script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "svg",
            "canvas", "form", "template", "select",
        ]
        .into_iter()
        .collect();

        while let Some(c) = chars.next() {
            if c == '<' {
                // Check if HTML comment <!-- ... -->
                if chars.clone().take(3).collect::<String>() == "!--" {
                    for _ in 0..3 {
                        chars.next();
                    }
                    let mut hyphens = 0;
                    for cc in chars.by_ref() {
                        if cc == '-' {
                            hyphens += 1;
                        } else if cc == '>' && hyphens >= 2 {
                            break;
                        } else {
                            hyphens = 0;
                        }
                    }
                    continue;
                }

                // Parse tag name
                let mut tag = String::new();
                let is_closing = chars.peek() == Some(&'/');
                if is_closing {
                    chars.next();
                }

                while let Some(&tc) = chars.peek() {
                    if tc.is_ascii_alphanumeric() {
                        tag.push(tc.to_ascii_lowercase());
                        chars.next();
                    } else {
                        break;
                    }
                }

                // Fast-forward to end of tag `>`
                for tc in chars.by_ref() {
                    if tc == '>' {
                        break;
                    }
                }

                if !is_closing && discard_tags.contains(tag.as_str()) {
                    // Skip until matching closing tag </tag>
                    let close_pattern = format!("</{}", tag);
                    let mut window = String::new();
                    while let Some(sc) = chars.next() {
                        window.push(sc.to_ascii_lowercase());
                        if window.ends_with(&close_pattern) {
                            for ec in chars.by_ref() {
                                if ec == '>' {
                                    break;
                                }
                            }
                            break;
                        }
                    }
                    continue;
                }

                // Block tag transitions
                match tag.as_str() {
                    "h1" => out.push_str("\n\n# "),
                    "h2" => out.push_str("\n\n## "),
                    "h3" => out.push_str("\n\n### "),
                    "h4" | "h5" | "h6" => out.push_str("\n\n#### "),
                    "p" | "div" | "section" | "article" | "blockquote" => out.push_str("\n\n"),
                    "li" => out.push_str("\n- "),
                    "br" => out.push('\n'),
                    "hr" => out.push_str("\n---\n"),
                    _ => {}
                }
            } else if c == '&' {
                // Parse entity
                let mut entity = String::new();
                let mut valid = false;
                for _ in 0..10 {
                    if let Some(&ec) = chars.peek() {
                        if ec == ';' {
                            chars.next();
                            valid = true;
                            break;
                        }
                        if ec.is_whitespace() || ec == '<' || ec == '>' || ec == '&' {
                            break;
                        }
                        chars.next();
                        entity.push(ec);
                    } else {
                        break;
                    }
                }
                if valid {
                    out.push_str(&Self::resolve_entity(&entity));
                } else {
                    out.push('&');
                    out.push_str(&entity);
                }
            } else {
                out.push(c);
            }
        }

        Self::normalize_whitespace(&out)
    }

    fn resolve_entity(e: &str) -> String {
        match e {
            "quot" => "\"".to_string(),
            "amp" => "&".to_string(),
            "lt" => "<".to_string(),
            "gt" => ">".to_string(),
            "apos" | "#39" => "'".to_string(),
            "nbsp" => " ".to_string(),
            "mdash" | "#8212" => "—".to_string(),
            "ndash" | "#8211" => "–".to_string(),
            "hellip" => "…".to_string(),
            "copy" => "©".to_string(),
            other if other.starts_with('#') => {
                let code_str = if other.starts_with("#x") || other.starts_with("#X") {
                    u32::from_str_radix(&other[2..], 16).ok()
                } else {
                    other[1..].parse::<u32>().ok()
                };
                if let Some(code) = code_str
                    && let Some(ch) = char::from_u32(code)
                {
                    return ch.to_string();
                }
                format!("&{};", other)
            }
            other => format!("&{};", other),
        }
    }

    pub fn decode_entities(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '&' {
                let mut ent = String::new();
                let mut found = false;
                while let Some(&ec) = chars.peek() {
                    chars.next();
                    if ec == ';' {
                        found = true;
                        break;
                    }
                    ent.push(ec);
                    if ent.len() > 10 {
                        break;
                    }
                }
                if found {
                    out.push_str(&Self::resolve_entity(&ent));
                } else {
                    out.push('&');
                    out.push_str(&ent);
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    fn normalize_whitespace(text: &str) -> String {
        let mut res = String::new();
        let mut consecutive_newlines = 0;

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if consecutive_newlines < 2 && !res.is_empty() {
                    res.push('\n');
                    consecutive_newlines += 1;
                }
            } else {
                if !res.is_empty() && consecutive_newlines == 0 {
                    res.push('\n');
                }
                let mut line_buf = String::new();
                for word in trimmed.split_whitespace() {
                    if !line_buf.is_empty() {
                        line_buf.push(' ');
                    }
                    line_buf.push_str(word);
                }
                res.push_str(&line_buf);
                consecutive_newlines = 0;
            }
        }
        res.trim().to_string()
    }
}

/// Execute an HTTP GET via system `curl` with headers and timeout, verifying exit status.
pub(crate) fn curl_fetch(url: &str, headers: &[(&str, &str)], timeout_secs: u64) -> Result<String> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sSL", "--max-time", &timeout_secs.to_string()]);
    for (k, v) in headers {
        cmd.args(["-H", &format!("{}: {}", k, v)]);
    }
    cmd.arg(url);

    let output = cmd
        .output()
        .with_context(|| format!("Failed to execute curl request for {}", url))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        bail!("HTTP request failed for {}: {}", url, err.trim());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Jina Reader API client (`https://r.jina.ai/<url>`) for dynamic SPA rendering.
pub struct JinaReaderClient;

impl JinaReaderClient {
    pub fn fetch(url: &str, api_key: Option<&str>, timeout_secs: u64) -> Result<ExtractedDocument> {
        let jina_url = format!("https://r.jina.ai/{}", url);
        let auth_header = api_key.map(|k| format!("Bearer {}", k));
        let mut headers = vec![
            ("Accept", "text/plain"),
            ("X-Return-Format", "markdown"),
            ("X-Timeout", "20"),
        ];
        if let Some(auth) = &auth_header {
            headers.push(("Authorization", auth.as_str()));
        }

        let body = curl_fetch(&jina_url, &headers, timeout_secs)?
            .trim()
            .to_string();
        if body.is_empty() {
            bail!("Jina Reader returned empty response for {}", url);
        }

        // Try extracting first title line: Title: <title> or # <title>
        let mut title = String::new();
        for line in body.lines().take(10) {
            let trimmed = line.trim();
            if let Some(rest) = trimmed
                .strip_prefix("Title:")
                .or_else(|| trimmed.strip_prefix("# "))
            {
                title = rest.trim().to_string();
                break;
            }
        }

        let word_count = body.split_whitespace().count();

        Ok(ExtractedDocument {
            url: url.to_string(),
            title,
            markdown: body,
            word_count,
            is_likely_spa: false,
        })
    }
}

/// Unified web scraper with local parsing and automatic Jina Reader fallback for SPAs.
pub struct WebScraper {
    options: ScrapeOptions,
}

impl WebScraper {
    pub fn new(options: ScrapeOptions) -> Self {
        Self { options }
    }

    pub fn scrape(&self, url: &str) -> Result<ExtractedDocument> {
        // If offline mode is enabled via environment, check fixture
        if std::env::var("LEMMA_OFFLINE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let fixture_path = std::path::Path::new("tests/fixtures/sample_article.html");
            if fixture_path.exists() {
                let html = std::fs::read_to_string(fixture_path)?;
                return Ok(HtmlExtractor::extract(&html, url));
            }
        }

        if self.options.use_jina {
            return JinaReaderClient::fetch(
                url,
                self.options.jina_api_key.as_deref(),
                self.options.timeout_secs,
            );
        }

        // 1. Fetch raw HTML via system curl (zero dependencies)
        let headers = [
            ("User-Agent", self.options.user_agent.as_str()),
            (
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            ),
            ("Accept-Language", "en-US,en;q=0.9"),
        ];
        let raw_html = curl_fetch(url, &headers, self.options.timeout_secs)?;
        let mut doc = HtmlExtractor::extract(&raw_html, url);

        // 2. Fallback check: if extracted text is minimal or SPA detected, try Jina Reader
        if (doc.is_likely_spa || doc.markdown.len() < 120)
            && !self.options.use_jina
            && let Ok(jina_doc) = JinaReaderClient::fetch(
                url,
                self.options.jina_api_key.as_deref(),
                self.options.timeout_secs,
            )
            && jina_doc.word_count > doc.word_count
        {
            doc = jina_doc;
        }

        // 3. Truncate safely on unicode character boundary
        let char_count = doc.markdown.chars().count();
        if char_count > self.options.max_chars {
            doc.markdown = doc.markdown.chars().take(self.options.max_chars).collect();
        }

        Ok(doc)
    }
}

/// Data source implementation allowing web URLs to feed directly into Lemma classifiers.
pub struct WebSource {
    pub urls: Vec<String>,
    pub scraper: WebScraper,
}

impl WebSource {
    pub fn new(urls: Vec<String>, options: ScrapeOptions) -> Self {
        Self {
            urls,
            scraper: WebScraper::new(options),
        }
    }
}

impl DataSource for WebSource {
    fn name(&self) -> &str {
        "WebSource"
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        use rayon::prelude::*;

        let items: Vec<DataItem> = self
            .urls
            .par_iter()
            .filter_map(|url| match self.scraper.scrape(url) {
                Ok(doc) => {
                    let state = serde_json::json!({
                        "title": doc.title,
                        "text": doc.markdown,
                        "url": doc.url,
                        "words": doc.word_count,
                    });
                    Some(DataItem::new(url.clone(), state, url.clone()))
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to scrape {}: {}", url, e);
                    None
                }
            })
            .collect();

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_to_markdown_basic_tags() {
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Page - News Portal</title>
                <script>console.log("ignore me");</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <header><nav><a href="/">Home</a></nav></header>
                <h1>Breaking News Headline</h1>
                <p>This is the first paragraph with &amp; entity and &quot;quotes&quot;.</p>
                <ul>
                    <li>First bullet</li>
                    <li>Second bullet</li>
                </ul>
                <footer>Copyright 2026</footer>
            </body>
            </html>
        "#;

        let doc = HtmlExtractor::extract(html, "https://example.com/test");
        assert_eq!(doc.title, "Test Page");
        assert!(doc.markdown.contains("# Breaking News Headline"));
        assert!(
            doc.markdown
                .contains("This is the first paragraph with & entity and \"quotes\".")
        );
        assert!(doc.markdown.contains("- First bullet"));
        assert!(doc.markdown.contains("- Second bullet"));
        assert!(!doc.markdown.contains("ignore me"));
        assert!(!doc.markdown.contains("color: red"));
        assert!(!doc.markdown.contains("Copyright 2026"));
    }

    #[test]
    fn test_html_entity_decoding() {
        let raw = "&quot;Hello &amp; World&quot; &mdash; &#39;test&#39; &#x20AC; &copy;";
        let decoded = HtmlExtractor::decode_entities(raw);
        assert_eq!(decoded, "\"Hello & World\" — 'test' € ©");
    }

    #[test]
    fn test_spa_detection() {
        let spa_html =
            "<html><body><div id=\"root\"></div><script src=\"bundle.js\"></script></body></html>";
        assert!(HtmlExtractor::detect_spa(spa_html));

        let static_html =
            "<html><body><article><p>Static article text here.</p></article></body></html>";
        assert!(!HtmlExtractor::detect_spa(static_html));
    }
}
