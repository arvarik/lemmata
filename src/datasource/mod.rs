pub mod directory;
pub mod file;
pub mod stdin;
pub mod traits;
pub mod web;

pub use directory::DirectorySource;
pub use file::FileSource;
pub use stdin::StdinSource;
pub use traits::{DataItem, DataSource, MemorySource};
pub use web::{ExtractedDocument, HtmlExtractor, ScrapeOptions, WebScraper, WebSource};
