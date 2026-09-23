use std::collections::BTreeMap;
use std::process::Command;
use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::traits::{DataItem, DataSource};
use crate::model::schema::Question;

/// Hugging Face Datasets Server API response data structures
#[derive(Debug, Clone, Deserialize)]
pub struct HfFeatureType {
    pub dtype: Option<String>,
    #[serde(rename = "_type")]
    pub type_name: String,
    pub names: Option<Vec<String>>,
    pub num_classes: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfFeature {
    pub feature_idx: usize,
    pub name: String,
    #[serde(rename = "type")]
    pub feature_type: HfFeatureType,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfRowWrapper {
    pub row_idx: usize,
    pub row: serde_json::Value,
    #[serde(default)]
    pub truncated_cells: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfRowsResponse {
    pub features: Vec<HfFeature>,
    pub rows: Vec<HfRowWrapper>,
    pub num_rows_total: usize,
    pub num_rows_per_page: usize,
    #[serde(default)]
    pub partial: bool,
}

#[derive(Debug, Clone)]
pub struct DiscoveredColumns {
    pub text_col: String,
    pub paired_text_col: Option<String>,
    pub label_col: Option<String>,
    pub id_col: Option<String>,
    pub class_labels: Option<Vec<String>>,
}

/// Hugging Face Dataset Ingestion Connector
pub struct HuggingFaceSource {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub limit: usize,
    pub offset: usize,
    pub token: Option<String>,
    pub custom_text_col: Option<String>,
    pub custom_label_col: Option<String>,
}

impl HuggingFaceSource {
    pub fn new(dataset: impl Into<String>) -> Self {
        Self {
            dataset: dataset.into(),
            config: "default".into(),
            split: "train".into(),
            limit: 50,
            offset: 0,
            token: std::env::var("HF_TOKEN").ok(),
            custom_text_col: None,
            custom_label_col: None,
        }
    }

    pub fn with_config(mut self, config: impl Into<String>) -> Self {
        self.config = config.into();
        self
    }

    pub fn with_split(mut self, split: impl Into<String>) -> Self {
        self.split = split.into();
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    pub fn with_custom_columns(mut self, text_col: Option<String>, label_col: Option<String>) -> Self {
        self.custom_text_col = text_col;
        self.custom_label_col = label_col;
        self
    }

    /// Resolve legacy dataset aliases to canonical Hugging Face namespaces
    pub fn resolve_canonical_dataset(name: &str) -> String {
        if name.contains('/') {
            return name.to_string();
        }
        match name.to_lowercase().as_str() {
            "ag_news" => "fancyzhx/ag_news".to_string(),
            "imdb" => "stanfordnlp/imdb".to_string(),
            "banking77" => "PolyAI/banking77".to_string(),
            "rotten_tomatoes" => "cornell-movie-review-data/rotten_tomatoes".to_string(),
            "tweet_eval" => "cardiffnlp/tweet_eval".to_string(),
            _ => name.to_string(),
        }
    }

    /// Auto-detect text, label, and ID columns from the features schema
    pub fn discover_columns(&self, features: &[HfFeature]) -> Result<DiscoveredColumns> {
        const CANDIDATE_TEXT: &[&str] = &[
            "text", "sentence", "content", "body", "review", "premise",
            "comment", "comment_text", "document", "article", "query", "question"
        ];
        const CANDIDATE_LABEL: &[&str] = &[
            "label", "target", "intent", "category", "class", "stars",
            "sentiment", "topic", "score", "ground_truth"
        ];
        const CANDIDATE_ID: &[&str] = &["id", "item_id", "idx", "row_id", "uuid"];

        let mut discovered_label = self.custom_label_col.clone();
        let mut class_labels = None;
        let mut text_col = self.custom_text_col.clone();
        let mut paired_text_col = None;
        let mut id_col = None;

        // 1. Label Column & Class Labels from ClassLabel type
        if let Some(ref custom_name) = discovered_label {
            if let Some(feat) = features.iter().find(|f| f.name.eq_ignore_ascii_case(custom_name)) {
                class_labels = feat.feature_type.names.clone();
            }
        } else {
            for feat in features {
                if feat.feature_type.type_name == "ClassLabel" {
                    discovered_label = Some(feat.name.clone());
                    class_labels = feat.feature_type.names.clone();
                    break;
                }
            }

            // Fallback label name match
            if discovered_label.is_none() {
                for candidate in CANDIDATE_LABEL {
                    if let Some(feat) = features.iter().find(|f| f.name.eq_ignore_ascii_case(candidate)) {
                        discovered_label = Some(feat.name.clone());
                        class_labels = feat.feature_type.names.clone();
                        break;
                    }
                }
            }
        }

        // 2. Text Column Discovery
        if text_col.is_none() {
            // Check for NLI pairs (premise / hypothesis)
            let has_premise = features.iter().any(|f| f.name.eq_ignore_ascii_case("premise"));
            let has_hyp = features.iter().any(|f| f.name.eq_ignore_ascii_case("hypothesis"));
            if has_premise && has_hyp {
                text_col = Some("premise".into());
                paired_text_col = Some("hypothesis".into());
            }

            // Check for sentence pairs (sentence1 / sentence2)
            let has_s1 = features.iter().any(|f| f.name.eq_ignore_ascii_case("sentence1"));
            let has_s2 = features.iter().any(|f| f.name.eq_ignore_ascii_case("sentence2"));
            if has_s1 && has_s2 {
                text_col = Some("sentence1".into());
                paired_text_col = Some("sentence2".into());
            }
        }

        if text_col.is_none() {
            for candidate in CANDIDATE_TEXT {
                if let Some(feat) = features.iter().find(|f| f.name.to_lowercase() == *candidate) {
                    text_col = Some(feat.name.clone());
                    break;
                }
            }
        }

        // 3. ID Column Discovery
        for candidate in CANDIDATE_ID {
            if let Some(feat) = features.iter().find(|f| f.name.to_lowercase() == *candidate) {
                id_col = Some(feat.name.clone());
                break;
            }
        }

        let text_col = text_col.ok_or_else(|| {
            anyhow::anyhow!("Could not automatically determine text column. Specify text_col explicitly.")
        })?;

        Ok(DiscoveredColumns {
            text_col,
            paired_text_col,
            label_col: discovered_label,
            id_col,
            class_labels,
        })
    }

    /// Auto-generate aligned Lemma questions schema from dataset class labels
    pub fn generate_questions_schema(&self, cols: &DiscoveredColumns) -> BTreeMap<String, Question> {
        let mut questions = BTreeMap::new();
        if let Some(labels) = &cols.class_labels {
            let q = Question::new_choice(
                format!("Classify the input into one of the designated categories for {}", self.dataset),
                labels.clone(),
            );
            questions.insert("classification".to_string(), q);
        } else {
            let q = Question::new_choice(
                format!("Classify the input for {}", self.dataset),
                vec!["negative".into(), "positive".into()],
            );
            questions.insert("classification".to_string(), q);
        }
        questions
    }

    fn fetch_page(&self, dataset_id: &str, offset: usize, length: usize) -> Result<HfRowsResponse> {
        // Offline test fixture check
        if std::env::var("LEMMA_OFFLINE").map(|v| v == "1").unwrap_or(false) {
            let fixture_path = std::path::Path::new("tests/fixtures/hf_ag_news.json");
            if fixture_path.exists() {
                let content = std::fs::read_to_string(fixture_path)?;
                let mut resp: HfRowsResponse = serde_json::from_str(&content)?;
                if offset < resp.rows.len() {
                    resp.rows = resp.rows.into_iter().skip(offset).take(length).collect();
                } else {
                    resp.rows.clear();
                }
                return Ok(resp);
            }
        }

        let length = length.min(100);
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length={}",
            dataset_id, self.config, self.split, offset, length
        );

        let mut cmd = Command::new("curl");
        cmd.args([
            "-sSL",
            "--max-time", "20",
            "-H", "Accept: application/json",
            &url,
        ]);
        if let Some(token) = &self.token {
            cmd.args(["-H", &format!("Authorization: Bearer {}", token)]);
        }

        let output = cmd.output().context("Failed to execute curl request to HF Datasets Server")?;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            bail!("HF Datasets Server request failed: {}", err);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let resp: HfRowsResponse = serde_json::from_str(&stdout)
            .with_context(|| format!("Failed to parse HF response JSON: {}", stdout))?;
        Ok(resp)
    }

    /// Read all items and return them along with the inferred question schema
    pub fn read_items_with_schema(&self) -> Result<(Vec<DataItem>, BTreeMap<String, Question>)> {
        let canonical_id = Self::resolve_canonical_dataset(&self.dataset);
        let mut items = Vec::new();
        let mut current_offset = self.offset;
        let mut cols: Option<DiscoveredColumns> = None;

        while items.len() < self.limit {
            let batch_size = (self.limit - items.len()).min(100);
            let page = self.fetch_page(&canonical_id, current_offset, batch_size)?;
            if page.rows.is_empty() {
                break;
            }

            if cols.is_none() {
                cols = Some(self.discover_columns(&page.features)?);
            }
            let c = cols.as_ref().unwrap();
            let page_rows_count = page.rows.len();

            for row_wrapper in page.rows {
                let row_val = row_wrapper.row;

                // 1. Build item ID
                let item_id = if let Some(id_field) = &c.id_col {
                    row_val.get(id_field)
                        .filter(|v| !v.is_null())
                        .map(|v| match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string().trim_matches('"').to_string(),
                        })
                        .unwrap_or_else(|| format!("{}:{}:{}", canonical_id, self.split, row_wrapper.row_idx))
                } else {
                    format!("{}:{}:{}", canonical_id, self.split, row_wrapper.row_idx)
                };

                // 2. Build item State payload
                let state = if let Some(p2) = &c.paired_text_col {
                    serde_json::json!({
                        &c.text_col: row_val.get(&c.text_col).cloned().unwrap_or_default(),
                        p2: row_val.get(p2).cloned().unwrap_or_default(),
                    })
                } else if let Some(txt_val) = row_val.get(&c.text_col) {
                    txt_val.clone()
                } else {
                    row_val.clone()
                };

                // 3. Resolve Ground Truth label
                let mut ground_truth = None;
                if let Some(label_field) = &c.label_col {
                    if let Some(raw_label) = row_val.get(label_field) {
                        let mut gt_map = BTreeMap::new();
                        if let Some(int_idx) = raw_label.as_u64() {
                            if let Some(names) = &c.class_labels {
                                if let Some(label_str) = names.get(int_idx as usize) {
                                    gt_map.insert("classification".to_string(), serde_json::Value::String(label_str.clone()));
                                } else {
                                    gt_map.insert("classification".to_string(), serde_json::Value::Number(int_idx.into()));
                                }
                            } else {
                                gt_map.insert("classification".to_string(), serde_json::Value::Number(int_idx.into()));
                            }
                        } else if let Some(s) = raw_label.as_str() {
                            gt_map.insert("classification".to_string(), serde_json::Value::String(s.to_string()));
                        }
                        if !gt_map.is_empty() {
                            ground_truth = Some(gt_map);
                        }
                    }
                }

                let source_uri = format!("hf://{}/{}/{}", canonical_id, self.config, self.split);
                let mut item = DataItem::new(item_id, state, source_uri);
                if let Some(gt) = ground_truth {
                    item = item.with_ground_truth(gt);
                }
                items.push(item);

                if items.len() >= self.limit {
                    break;
                }
            }

            current_offset += page_rows_count;
            if current_offset >= page.num_rows_total || page_rows_count == 0 {
                break;
            }
        }

        let schema = if let Some(c) = &cols {
            self.generate_questions_schema(c)
        } else {
            BTreeMap::new()
        };

        Ok((items, schema))
    }
}

impl DataSource for HuggingFaceSource {
    fn name(&self) -> &str {
        "HuggingFaceSource"
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        let (items, _) = self.read_items_with_schema()?;
        Ok(items)
    }
}

/// Hacker News Live Ingestion Connector
pub struct HackerNewsSource {
    pub category: String,
    pub limit: usize,
}

impl HackerNewsSource {
    pub fn new(category: impl Into<String>, limit: usize) -> Self {
        Self {
            category: category.into(),
            limit,
        }
    }
}

impl DataSource for HackerNewsSource {
    fn name(&self) -> &str {
        "HackerNewsSource"
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        // Offline test fixture check
        if std::env::var("LEMMA_OFFLINE").map(|v| v == "1").unwrap_or(false) {
            let fixture_path = std::path::Path::new("tests/fixtures/hn_item.json");
            if fixture_path.exists() {
                let content = std::fs::read_to_string(fixture_path)?;
                let val: serde_json::Value = serde_json::from_str(&content)?;
                let item = DataItem::new(
                    "hn:39000000",
                    val,
                    "hn:top",
                );
                return Ok(vec![item]);
            }
        }

        let lowered = self.category.to_lowercase();
        let feed_name = match lowered.as_str() {
            "top" => "topstories",
            "new" => "newstories",
            "best" => "beststories",
            "ask" => "askstories",
            "show" => "showstories",
            other => other,
        };

        let list_url = format!("https://hacker-news.firebaseio.com/v0/{}.json", feed_name);
        let output = Command::new("curl")
            .args(["-sSL", "--max-time", "10", &list_url])
            .output()
            .context("Failed to query Hacker News story list")?;

        let ids: Vec<u64> = serde_json::from_slice(&output.stdout)
            .unwrap_or_default();

        let mut items = Vec::new();
        for id in ids.into_iter().take(self.limit) {
            let item_url = format!("https://hacker-news.firebaseio.com/v0/item/{}.json", id);
            let item_output = Command::new("curl")
                .args(["-sSL", "--max-time", "5", &item_url])
                .output();

            if let Ok(out) = item_output {
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                    if !val.is_null() {
                        let title = val.get("title").and_then(|t| t.as_str()).unwrap_or_default();
                        let text = val.get("text").and_then(|t| t.as_str()).unwrap_or(title);
                        let clean_text = super::web::HtmlExtractor::decode_entities(text);

                        let state = serde_json::json!({
                            "title": title,
                            "text": clean_text,
                            "by": val.get("by").and_then(|b| b.as_str()).unwrap_or_default(),
                            "score": val.get("score").and_then(|s| s.as_u64()).unwrap_or(0),
                            "url": val.get("url").and_then(|u| u.as_str()).unwrap_or_default(),
                            "type": val.get("type").and_then(|t| t.as_str()).unwrap_or("story"),
                        });

                        items.push(DataItem::new(
                            format!("hn:{}", id),
                            state,
                            format!("hn:{}", self.category),
                        ));
                    }
                }
            }
        }

        Ok(items)
    }
}

/// Wikipedia Article Summary Ingestion Connector
pub struct WikipediaSource {
    pub article_or_random: Option<String>,
    pub count: usize,
}

impl WikipediaSource {
    pub fn new(article: Option<String>, count: usize) -> Self {
        Self {
            article_or_random: article,
            count,
        }
    }
}

impl DataSource for WikipediaSource {
    fn name(&self) -> &str {
        "WikipediaSource"
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        // Offline test fixture check
        if std::env::var("LEMMA_OFFLINE").map(|v| v == "1").unwrap_or(false) {
            let fixture_path = std::path::Path::new("tests/fixtures/wikipedia_rust.json");
            if fixture_path.exists() {
                let content = std::fs::read_to_string(fixture_path)?;
                let val: serde_json::Value = serde_json::from_str(&content)?;
                let item = DataItem::new(
                    "wiki:29414838",
                    val,
                    "wiki:Rust_(programming_language)",
                );
                return Ok(vec![item]);
            }
        }

        let user_agent = "LemmaBot/0.1 (https://github.com/arvarik/lemma; pair@lemma.ai)";
        let mut items = Vec::new();

        if let Some(title) = &self.article_or_random {
            let enc = title.replace(' ', "_");
            let url = format!("https://en.wikipedia.org/api/rest_v1/page/summary/{}", enc);
            let output = Command::new("curl")
                .args(["-sSL", "--max-time", "10", "-A", user_agent, &url])
                .output()
                .context("Failed to fetch Wikipedia article summary")?;

            if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                let page_id = val.get("pageid").map(|p| p.to_string()).unwrap_or_else(|| title.clone());
                let title_str = val.get("title").and_then(|t| t.as_str()).unwrap_or(title);
                let extract = val.get("extract").and_then(|e| e.as_str()).unwrap_or_default();
                let desc = val.get("description").and_then(|d| d.as_str()).unwrap_or_default();

                let state = serde_json::json!({
                    "title": title_str,
                    "text": extract,
                    "description": desc,
                });

                items.push(DataItem::new(
                    format!("wiki:{}", page_id),
                    state,
                    format!("wiki:{}", enc),
                ));
            }
        } else {
            // Random summaries
            for _ in 0..self.count {
                let url = "https://en.wikipedia.org/api/rest_v1/page/random/summary";
                let output = Command::new("curl")
                    .args(["-sSL", "--max-time", "10", "-A", user_agent, url])
                    .output();

                if let Ok(out) = output {
                    if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                        if let Some(title) = val.get("title").and_then(|t| t.as_str()) {
                            let page_id = val.get("pageid").map(|p| p.to_string()).unwrap_or_else(|| title.to_string());
                            let extract = val.get("extract").and_then(|e| e.as_str()).unwrap_or_default();
                            let desc = val.get("description").and_then(|d| d.as_str()).unwrap_or_default();

                            let state = serde_json::json!({
                                "title": title,
                                "text": extract,
                                "description": desc,
                            });

                            items.push(DataItem::new(
                                format!("wiki:{}", page_id),
                                state,
                                format!("wiki:{}", title),
                            ));
                        }
                    }
                }
            }
        }

        Ok(items)
    }
}

/// Reddit Subreddit Ingestion Connector
pub struct RedditSource {
    pub subreddit: String,
    pub limit: usize,
}

impl RedditSource {
    pub fn new(subreddit: impl Into<String>, limit: usize) -> Self {
        Self {
            subreddit: subreddit.into(),
            limit,
        }
    }
}

impl DataSource for RedditSource {
    fn name(&self) -> &str {
        "RedditSource"
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        // Desktop user agent to avoid Reddit CDN blocking
        let ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 (LemmaCLI/0.1)";
        let url = format!("https://www.reddit.com/r/{}/hot.json?limit={}", self.subreddit, self.limit);

        let output = Command::new("curl")
            .args(["-sSL", "--max-time", "10", "-A", ua, &url])
            .output()
            .context("Failed to query Reddit JSON API")?;

        let val: serde_json::Value = serde_json::from_slice(&output.stdout)
            .context("Failed to parse Reddit response as JSON (may be rate-limited or blocked)")?;

        if let Some(err) = val.get("error").or_else(|| val.get("message")) {
            bail!("Reddit API returned error: {}", err);
        }

        let mut items = Vec::new();
        if let Some(children) = val.get("data").and_then(|d| d.get("children")).and_then(|c| c.as_array()) {
            for child in children.iter().take(self.limit) {
                if let Some(post) = child.get("data") {
                    let id = post.get("id").and_then(|i| i.as_str()).unwrap_or_default();
                    let title = post.get("title").and_then(|t| t.as_str()).unwrap_or_default();
                    let selftext = post.get("selftext").and_then(|s| s.as_str()).unwrap_or_default();
                    let subreddit = post.get("subreddit").and_then(|s| s.as_str()).unwrap_or(&self.subreddit);
                    let score = post.get("score").and_then(|s| s.as_i64()).unwrap_or(0);
                    let flair = post.get("link_flair_text").and_then(|f| f.as_str()).unwrap_or_default();

                    let text = if selftext.is_empty() {
                        title.to_string()
                    } else {
                        format!("{}\n\n{}", title, selftext)
                    };

                    let state = serde_json::json!({
                        "title": title,
                        "text": text,
                        "subreddit": subreddit,
                        "flair": flair,
                        "score": score,
                    });

                    let mut gt = BTreeMap::new();
                    gt.insert("classification".to_string(), serde_json::Value::String(subreddit.to_string()));

                    items.push(
                        DataItem::new(format!("reddit:{}", id), state, format!("reddit:{}", self.subreddit))
                            .with_ground_truth(gt),
                    );
                }
            }
        }

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_dataset_resolution() {
        assert_eq!(HuggingFaceSource::resolve_canonical_dataset("ag_news"), "fancyzhx/ag_news");
        assert_eq!(HuggingFaceSource::resolve_canonical_dataset("imdb"), "stanfordnlp/imdb");
        assert_eq!(HuggingFaceSource::resolve_canonical_dataset("banking77"), "PolyAI/banking77");
        assert_eq!(HuggingFaceSource::resolve_canonical_dataset("custom/repo"), "custom/repo");
    }

    #[test]
    fn test_column_discovery_and_schema_generation() {
        let features = vec![
            HfFeature {
                feature_idx: 0,
                name: "text".to_string(),
                feature_type: HfFeatureType {
                    dtype: Some("string".to_string()),
                    type_name: "Value".to_string(),
                    names: None,
                    num_classes: None,
                },
            },
            HfFeature {
                feature_idx: 1,
                name: "label".to_string(),
                feature_type: HfFeatureType {
                    dtype: None,
                    type_name: "ClassLabel".to_string(),
                    names: Some(vec!["World".into(), "Sports".into(), "Business".into(), "Sci/Tech".into()]),
                    num_classes: Some(4),
                },
            },
        ];

        let source = HuggingFaceSource::new("fancyzhx/ag_news");
        let cols = source.discover_columns(&features).unwrap();
        assert_eq!(cols.text_col, "text");
        assert_eq!(cols.label_col, Some("label".to_string()));
        assert_eq!(cols.class_labels.as_ref().unwrap().len(), 4);

        let schema = source.generate_questions_schema(&cols);
        assert!(schema.contains_key("classification"));
        let q = &schema["classification"];
        let mut opts = q.render_options();
        opts.sort();
        assert_eq!(opts, vec!["Business", "Sci/Tech", "Sports", "World"]);
    }
}
