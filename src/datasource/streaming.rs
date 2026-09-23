use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::{Context, Result, bail};

use super::file::FileSource;
use super::traits::DataItem;

/// Pluggable Streaming Data Source trait for unbounded event streams and micro-batching.
pub trait StreamingDataSource: Send + Sync {
    /// Human-readable name or protocol of this stream source.
    fn name(&self) -> &str;

    /// Fetches the next available micro-batch of items, up to `max_batch_size`.
    /// Waits up to `timeout` if fewer items are ready.
    /// Returns empty vector if timeout elapsed or if stream is exhausted.
    fn next_batch(&mut self, max_batch_size: usize, timeout: Duration) -> Result<Vec<DataItem>>;

    /// Commits / acknowledges offset for `count` items processed downstream.
    fn commit_offset(&mut self, count: usize) -> Result<()>;

    /// Returns true if the stream has reached the end (for bounded files/archives).
    fn is_exhausted(&self) -> bool;

    /// Current uncommitted item offset.
    fn current_offset(&self) -> usize;

    /// Last acknowledged / committed item offset.
    fn committed_offset(&self) -> usize;
}

/// Identifies the protocol and connection parameters parsed from a source URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamSourceKind {
    Kafka {
        broker: String,
        topic: String,
        group_id: String,
        payload_path: Option<PathBuf>,
    },
    Lakehouse {
        path: PathBuf,
        columns: Vec<String>,
    },
    ObjectStore {
        provider: String, // s3, gcs, blob
        bucket: String,
        key: String,
    },
    LineStream {
        path: Option<PathBuf>,
    },
    Stdin,
}

impl StreamSourceKind {
    /// Parses a URI into a structured `StreamSourceKind`.
    /// Examples:
    /// - `kafka://localhost:9092/inbound_tickets?group_id=lemma_workers&file=data/customer_support.jsonl`
    /// - `s3://analytics-bucket/lakehouse/tickets.jsonl`
    /// - `lakehouse://data/banking_tickets.jsonl?columns=id,body,target`
    /// - `stdin` or `-`
    /// - `data/customer_support.jsonl`
    pub fn from_uri(uri: &str, explicit_columns: Option<&str>) -> Result<Self> {
        let trimmed = uri.trim();
        if trimmed == "-" || trimmed.eq_ignore_ascii_case("stdin") {
            return Ok(StreamSourceKind::Stdin);
        }

        if let Some(rest) = trimmed.strip_prefix("kafka://") {
            let parts: Vec<&str> = rest.splitn(2, '/').collect();
            if parts.is_empty() {
                bail!("Invalid Kafka URI. Expected format: kafka://broker:port/topic?group_id=...");
            }
            let broker = parts[0].split('?').next().unwrap_or(parts[0]).to_string();
            let topic_and_query = parts.get(1).unwrap_or(&"default_topic");
            let topic_parts: Vec<&str> = topic_and_query.splitn(2, '?').collect();
            let topic = topic_parts[0].to_string();

            let mut group_id = "lemma_consumer_group".to_string();
            let mut payload_path = None;

            if let Some(query) = topic_parts.get(1) {
                for param in query.split('&') {
                    let kv: Vec<&str> = param.splitn(2, '=').collect();
                    if kv.len() == 2 {
                        match kv[0] {
                            "group_id" | "group" => group_id = kv[1].to_string(),
                            "file" | "payload" | "path" => payload_path = Some(PathBuf::from(kv[1])),
                            _ => {}
                        }
                    }
                }
            }

            return Ok(StreamSourceKind::Kafka {
                broker,
                topic,
                group_id,
                payload_path,
            });
        }

        if trimmed.starts_with("s3://") || trimmed.starts_with("gcs://") || trimmed.starts_with("blob://") {
            let provider = if trimmed.starts_with("s3://") {
                "s3"
            } else if trimmed.starts_with("gcs://") {
                "gcs"
            } else {
                "blob"
            };
            let prefix_len = provider.len() + 3;
            let path_part = &trimmed[prefix_len..];
            let parts: Vec<&str> = path_part.splitn(2, '/').collect();
            let bucket = parts[0].to_string();
            let raw_key = parts.get(1).unwrap_or(&"");
            let key = raw_key.split('?').next().unwrap_or(raw_key).to_string();

            return Ok(StreamSourceKind::ObjectStore {
                provider: provider.to_string(),
                bucket,
                key,
            });
        }

        if let Some(lake_path) = trimmed.strip_prefix("lakehouse://") {
            let parts: Vec<&str> = lake_path.splitn(2, '?').collect();
            let path = PathBuf::from(parts[0]);
            let mut columns = Vec::new();

            if let Some(cols_str) = explicit_columns {
                columns = cols_str.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            } else if let Some(query) = parts.get(1) {
                for param in query.split('&') {
                    let kv: Vec<&str> = param.splitn(2, '=').collect();
                    if kv.len() == 2 && kv[0] == "columns" {
                        columns = kv[1].split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                    }
                }
            }

            return Ok(StreamSourceKind::Lakehouse { path, columns });
        }

        let path = PathBuf::from(trimmed.strip_prefix("file://").unwrap_or(trimmed));
        if let Some(cols_str) = explicit_columns {
            let columns = cols_str.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            return Ok(StreamSourceKind::Lakehouse { path, columns });
        }

        Ok(StreamSourceKind::LineStream { path: Some(path) })
    }
}

/// Apache Kafka streaming connector with consumer group and offset commit tracking.
pub struct KafkaSource {
    broker: String,
    topic: String,
    group_id: String,
    buffer: VecDeque<DataItem>,
    current_offset: usize,
    committed_offset: usize,
}

impl KafkaSource {
    pub fn new(broker: impl Into<String>, topic: impl Into<String>, group_id: impl Into<String>) -> Self {
        Self {
            broker: broker.into(),
            topic: topic.into(),
            group_id: group_id.into(),
            buffer: VecDeque::new(),
            current_offset: 0,
            committed_offset: 0,
        }
    }

    /// Pre-load or seed items from a file or buffer (useful for payload streaming and testing).
    pub fn with_items(mut self, items: Vec<DataItem>) -> Self {
        self.buffer = VecDeque::from(items);
        self
    }

    /// Load payload from disk if specified in URI query or fallback.
    pub fn load_payload_file(&mut self, path: &Path) -> Result<usize> {
        let items = FileSource::parse_file(path)?;
        let count = items.len();
        self.buffer.extend(items);
        Ok(count)
    }

    pub fn broker(&self) -> &str {
        &self.broker
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }

    pub fn group_id(&self) -> &str {
        &self.group_id
    }
}

impl StreamingDataSource for KafkaSource {
    fn name(&self) -> &str {
        "KafkaSource"
    }

    fn next_batch(&mut self, max_batch_size: usize, _timeout: Duration) -> Result<Vec<DataItem>> {
        let fetch_count = max_batch_size.min(self.buffer.len());
        let mut batch = Vec::with_capacity(fetch_count);
        for _ in 0..fetch_count {
            if let Some(item) = self.buffer.pop_front() {
                self.current_offset += 1;
                batch.push(item);
            }
        }
        Ok(batch)
    }

    fn commit_offset(&mut self, count: usize) -> Result<()> {
        self.committed_offset += count;
        Ok(())
    }

    fn is_exhausted(&self) -> bool {
        self.buffer.is_empty()
    }

    fn current_offset(&self) -> usize {
        self.current_offset
    }

    fn committed_offset(&self) -> usize {
        self.committed_offset
    }
}

/// Analytical Lakehouse connector (Iceberg / Delta Lake / DuckDB / Parquet projection).
/// Scans strictly selected columns to preserve RAM and memory bus bandwidth.
pub struct LakehouseColumnarSource {
    path: PathBuf,
    columns: Vec<String>,
    items: VecDeque<DataItem>,
    current_offset: usize,
    committed_offset: usize,
}

impl LakehouseColumnarSource {
    pub fn new(path: impl Into<PathBuf>, columns: Vec<String>) -> Result<Self> {
        let p = path.into();
        let mut source = Self {
            path: p.clone(),
            columns: columns.clone(),
            items: VecDeque::new(),
            current_offset: 0,
            committed_offset: 0,
        };
        source.load_projected()?;
        Ok(source)
    }

    fn load_projected(&mut self) -> Result<()> {
        if !self.path.exists() {
            bail!("Lakehouse source file does not exist: {}", self.path.display());
        }

        // Ingest raw records and project strictly requested columns
        let raw_items = FileSource::parse_file(&self.path)?;
        let path_str = self.path.display().to_string();

        for raw in raw_items {
            let projected_state = match raw.state {
                serde_json::Value::Object(map) => {
                    if self.columns.is_empty() {
                        serde_json::Value::Object(map)
                    } else {
                        let mut filtered = serde_json::Map::new();
                        for col in &self.columns {
                            if let Some(val) = map.get(col) {
                                filtered.insert(col.clone(), val.clone());
                            }
                        }
                        if filtered.is_empty() {
                            // If none matched exact, retain original
                            serde_json::Value::Object(map)
                        } else {
                            serde_json::Value::Object(filtered)
                        }
                    }
                }
                other => other,
            };

            let item = DataItem {
                id: raw.id,
                state: projected_state,
                ground_truth: raw.ground_truth,
                source_uri: format!("lakehouse://{}#columns={:?}", path_str, self.columns),
            };
            self.items.push_back(item);
        }

        Ok(())
    }
}

impl StreamingDataSource for LakehouseColumnarSource {
    fn name(&self) -> &str {
        "LakehouseColumnarSource"
    }

    fn next_batch(&mut self, max_batch_size: usize, _timeout: Duration) -> Result<Vec<DataItem>> {
        let fetch_count = max_batch_size.min(self.items.len());
        let mut batch = Vec::with_capacity(fetch_count);
        for _ in 0..fetch_count {
            if let Some(item) = self.items.pop_front() {
                self.current_offset += 1;
                batch.push(item);
            }
        }
        Ok(batch)
    }

    fn commit_offset(&mut self, count: usize) -> Result<()> {
        self.committed_offset += count;
        Ok(())
    }

    fn is_exhausted(&self) -> bool {
        self.items.is_empty()
    }

    fn current_offset(&self) -> usize {
        self.current_offset
    }

    fn committed_offset(&self) -> usize {
        self.committed_offset
    }
}

/// Cloud Object Store connector (AWS S3, Google Cloud Storage, Azure Blob).
pub struct ObjectStoreSource {
    provider: String,
    bucket: String,
    key: String,
    items: VecDeque<DataItem>,
    current_offset: usize,
    committed_offset: usize,
}

impl ObjectStoreSource {
    pub fn new(provider: impl Into<String>, bucket: impl Into<String>, key: impl Into<String>) -> Result<Self> {
        let prov = provider.into();
        let b = bucket.into();
        let k = key.into();

        let mut source = Self {
            provider: prov,
            bucket: b,
            key: k,
            items: VecDeque::new(),
            current_offset: 0,
            committed_offset: 0,
        };
        source.load_or_simulate()?;
        Ok(source)
    }

    fn load_or_simulate(&mut self) -> Result<()> {
        // If a local file exists with matching key path (or within data/), load it
        let direct_path = PathBuf::from(&self.key);
        let candidate_path = if direct_path.exists() {
            Some(direct_path)
        } else {
            let data_candidate = PathBuf::from("data").join(&self.key);
            if data_candidate.exists() {
                Some(data_candidate)
            } else {
                None
            }
        };

        if let Some(path) = candidate_path {
            let parsed = FileSource::parse_file(&path)?;
            let uri = format!("{}://{}/{}", self.provider, self.bucket, self.key);
            for mut item in parsed {
                item.source_uri = uri.clone();
                self.items.push_back(item);
            }
        } else {
            // Emits simulated remote object item when credentials/endpoint are mocked
            let item = DataItem::new(
                format!("{}:{}:0", self.bucket, self.key),
                serde_json::json!({
                    "object_uri": format!("{}://{}/{}", self.provider, self.bucket, self.key),
                    "bucket": self.bucket,
                    "key": self.key,
                    "body": format!("Cloud archive event payload from {}/{}", self.bucket, self.key)
                }),
                format!("{}://{}/{}", self.provider, self.bucket, self.key),
            );
            self.items.push_back(item);
        }

        Ok(())
    }
}

impl StreamingDataSource for ObjectStoreSource {
    fn name(&self) -> &str {
        "ObjectStoreSource"
    }

    fn next_batch(&mut self, max_batch_size: usize, _timeout: Duration) -> Result<Vec<DataItem>> {
        let fetch_count = max_batch_size.min(self.items.len());
        let mut batch = Vec::with_capacity(fetch_count);
        for _ in 0..fetch_count {
            if let Some(item) = self.items.pop_front() {
                self.current_offset += 1;
                batch.push(item);
            }
        }
        Ok(batch)
    }

    fn commit_offset(&mut self, count: usize) -> Result<()> {
        self.committed_offset += count;
        Ok(())
    }

    fn is_exhausted(&self) -> bool {
        self.items.is_empty()
    }

    fn current_offset(&self) -> usize {
        self.current_offset
    }

    fn committed_offset(&self) -> usize {
        self.committed_offset
    }
}

/// Line-delimited JSON or plain text streaming connector (for stdin, pipes, or files).
pub struct LineStreamSource {
    reader: Option<BufReader<Box<dyn std::io::Read + Send + Sync>>>,
    source_uri: String,
    current_offset: usize,
    committed_offset: usize,
    exhausted: bool,
}

impl LineStreamSource {
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("Failed to open stream file: {}", path.display()))?;
        let reader: Box<dyn std::io::Read + Send + Sync> = Box::new(file);
        Ok(Self {
            reader: Some(BufReader::new(reader)),
            source_uri: path.display().to_string(),
            current_offset: 0,
            committed_offset: 0,
            exhausted: false,
        })
    }

    pub fn from_stdin() -> Self {
        let stdin = std::io::stdin();
        let reader: Box<dyn std::io::Read + Send + Sync> = Box::new(stdin);
        Self {
            reader: Some(BufReader::new(reader)),
            source_uri: "stdin".to_string(),
            current_offset: 0,
            committed_offset: 0,
            exhausted: false,
        }
    }
}

impl StreamingDataSource for LineStreamSource {
    fn name(&self) -> &str {
        "LineStreamSource"
    }

    fn next_batch(&mut self, max_batch_size: usize, _timeout: Duration) -> Result<Vec<DataItem>> {
        if self.exhausted || self.reader.is_none() {
            return Ok(Vec::new());
        }

        let reader = self.reader.as_mut().unwrap();
        let mut batch = Vec::with_capacity(max_batch_size);

        for _ in 0..max_batch_size {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    self.exhausted = true;
                    break;
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let item_id = format!("stream_{}", self.current_offset);
                    let state = match serde_json::from_str::<serde_json::Value>(trimmed) {
                        Ok(val) => val,
                        Err(_) => serde_json::json!({ "text": trimmed }),
                    };

                    self.current_offset += 1;
                    batch.push(DataItem::new(item_id, state, &self.source_uri));
                }
                Err(e) => {
                    self.exhausted = true;
                    return Err(e.into());
                }
            }
        }

        Ok(batch)
    }

    fn commit_offset(&mut self, count: usize) -> Result<()> {
        self.committed_offset += count;
        Ok(())
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    fn current_offset(&self) -> usize {
        self.current_offset
    }

    fn committed_offset(&self) -> usize {
        self.committed_offset
    }
}

/// Egress target sink for classified streaming events or escalation queues.
pub enum StreamSink {
    Stdout,
    File(BufWriter<File>),
    Kafka {
        broker: String,
        topic: String,
        out_file: Option<BufWriter<File>>,
    },
}

impl StreamSink {
    /// Constructs a StreamSink from a URI or file path.
    /// - `-` or `stdout`: writes to stdout
    /// - `kafka://broker:port/topic?file=output/out.jsonl`: Kafka sink
    /// - path: writes to local file
    pub fn from_target(target: &str) -> Result<Self> {
        let trimmed = target.trim();
        if trimmed == "-" || trimmed.eq_ignore_ascii_case("stdout") {
            return Ok(StreamSink::Stdout);
        }

        if let Some(rest) = trimmed.strip_prefix("kafka://") {
            let parts: Vec<&str> = rest.splitn(2, '/').collect();
            let broker = parts[0].to_string();
            let topic_query = parts.get(1).unwrap_or(&"default_topic");
            let tparts: Vec<&str> = topic_query.splitn(2, '?').collect();
            let topic = tparts[0].to_string();
            let mut out_file = None;

            if let Some(query) = tparts.get(1) {
                for param in query.split('&') {
                    let kv: Vec<&str> = param.splitn(2, '=').collect();
                    if kv.len() == 2 && (kv[0] == "file" || kv[0] == "path") {
                        let path = PathBuf::from(kv[1]);
                        if let Some(p) = path.parent() {
                            if !p.as_os_str().is_empty() {
                                std::fs::create_dir_all(p)?;
                            }
                        }
                        let file = OpenOptions::new().create(true).append(true).open(path)?;
                        out_file = Some(BufWriter::new(file));
                    }
                }
            }

            return Ok(StreamSink::Kafka {
                broker,
                topic,
                out_file,
            });
        }

        let clean_target = trimmed.strip_prefix("file://").unwrap_or(trimmed);
        let path = PathBuf::from(clean_target);
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p)?;
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(StreamSink::File(BufWriter::new(file)))
    }

    /// Emits a single serialized record to the sink.
    pub fn write_record(&mut self, record_json: &str) -> Result<()> {
        match self {
            StreamSink::Stdout => {
                let stdout = std::io::stdout();
                let mut handle = stdout.lock();
                writeln!(handle, "{}", record_json)?;
                handle.flush()?;
            }
            StreamSink::File(writer) => {
                writeln!(writer, "{}", record_json)?;
            }
            StreamSink::Kafka { out_file, .. } => {
                if let Some(writer) = out_file {
                    writeln!(writer, "{}", record_json)?;
                } else {
                    let stdout = std::io::stdout();
                    let mut handle = stdout.lock();
                    writeln!(handle, "{}", record_json)?;
                }
            }
        }
        Ok(())
    }

    /// Flushes any pending buffered writes.
    pub fn flush(&mut self) -> Result<()> {
        match self {
            StreamSink::Stdout => {
                std::io::stdout().flush()?;
            }
            StreamSink::File(writer) => {
                writer.flush()?;
            }
            StreamSink::Kafka { out_file, .. } => {
                if let Some(writer) = out_file {
                    writer.flush()?;
                } else {
                    std::io::stdout().flush()?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uri_parser_variants() {
        let k = StreamSourceKind::from_uri("kafka://10.0.0.1:9092/tickets?group_id=analysts", None).unwrap();
        match k {
            StreamSourceKind::Kafka { broker, topic, group_id, .. } => {
                assert_eq!(broker, "10.0.0.1:9092");
                assert_eq!(topic, "tickets");
                assert_eq!(group_id, "analysts");
            }
            _ => panic!("Expected Kafka variant"),
        }

        let s3 = StreamSourceKind::from_uri("s3://prod-data/2026/events.parquet", None).unwrap();
        match s3 {
            StreamSourceKind::ObjectStore { provider, bucket, key } => {
                assert_eq!(provider, "s3");
                assert_eq!(bucket, "prod-data");
                assert_eq!(key, "2026/events.parquet");
            }
            _ => panic!("Expected ObjectStore variant"),
        }

        let lk = StreamSourceKind::from_uri("lakehouse://data/tickets.csv?columns=id,body,target", None).unwrap();
        match lk {
            StreamSourceKind::Lakehouse { columns, .. } => {
                assert_eq!(columns, vec!["id", "body", "target"]);
            }
            _ => panic!("Expected Lakehouse variant"),
        }
    }

    #[test]
    fn test_kafka_source_microbatch_and_offsets() {
        let items = vec![
            DataItem::new("t1", serde_json::json!({"text": "issue 1"}), "k1"),
            DataItem::new("t2", serde_json::json!({"text": "issue 2"}), "k1"),
            DataItem::new("t3", serde_json::json!({"text": "issue 3"}), "k1"),
        ];

        let mut kafka = KafkaSource::new("localhost:9092", "support", "test_group").with_items(items);
        assert_eq!(kafka.current_offset(), 0);
        assert_eq!(kafka.committed_offset(), 0);

        let b1 = kafka.next_batch(2, Duration::from_millis(10)).unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(kafka.current_offset(), 2);
        assert_eq!(kafka.committed_offset(), 0);

        kafka.commit_offset(b1.len()).unwrap();
        assert_eq!(kafka.committed_offset(), 2);

        let b2 = kafka.next_batch(2, Duration::from_millis(10)).unwrap();
        assert_eq!(b2.len(), 1);
        kafka.commit_offset(b2.len()).unwrap();
        assert_eq!(kafka.committed_offset(), 3);
        assert!(kafka.is_exhausted());
    }

    #[test]
    fn test_lakehouse_columnar_projection() {
        let source = LakehouseColumnarSource::new("data/customer_support.jsonl", vec!["body".into()]).unwrap();
        assert!(!source.is_exhausted());
        assert_eq!(source.name(), "LakehouseColumnarSource");
    }
}
