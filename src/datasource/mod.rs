pub mod directory;
pub mod fetch;
pub mod file;
pub mod stdin;
pub mod streaming;
pub mod traits;
pub mod web;

pub use directory::DirectorySource;
pub use fetch::{HackerNewsSource, HuggingFaceSource, RedditSource, WikipediaSource};
pub use file::FileSource;
pub use stdin::StdinSource;
pub use streaming::{
    KafkaSource, LakehouseColumnarSource, LineStreamSource, ObjectStoreSource, StreamSink,
    StreamSourceKind, StreamingDataSource,
};
pub use traits::{DataItem, DataSource, MemorySource};
pub use web::{ExtractedDocument, HtmlExtractor, JinaReaderClient, ScrapeOptions, WebScraper, WebSource};
