use std::path::PathBuf;
use anyhow::{Result, bail};
use walkdir::WalkDir;

use super::file::FileSource;
use super::traits::{DataItem, DataSource};

/// Ingests data from an entire directory (e.g. the gitignored `data/` folder).
pub struct DirectorySource {
    dir_path: PathBuf,
    recursive: bool,
    extensions: Vec<String>,
}

impl DirectorySource {
    pub fn new(dir_path: impl Into<PathBuf>) -> Self {
        Self {
            dir_path: dir_path.into(),
            recursive: true,
            extensions: vec![
                "json".into(),
                "jsonl".into(),
                "csv".into(),
                "txt".into(),
                "md".into(),
            ],
        }
    }

    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.extensions = extensions;
        self
    }
}

impl DataSource for DirectorySource {
    fn name(&self) -> &str {
        "DirectorySource"
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        let dir = &self.dir_path;
        if !dir.exists() {
            bail!("Directory does not exist: {}", dir.display());
        }
        if !dir.is_dir() {
            bail!("Path is not a directory: {}", dir.display());
        }

        let mut walker = WalkDir::new(dir);
        if !self.recursive {
            walker = walker.max_depth(1);
        }

        let mut all_items = Vec::new();
        let mut file_count = 0;

        for entry in walker
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !name.starts_with('.') && name != "target"
            })
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() {
                let matches_ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|ext| self.extensions.iter().any(|allowed| allowed.eq_ignore_ascii_case(ext)))
                    .unwrap_or(false);

                if matches_ext {
                    match FileSource::parse_file(path) {
                        Ok(items) => {
                            file_count += 1;
                            all_items.extend(items);
                        }
                        Err(err) => {
                            eprintln!("⚠️  Skipping file {}: {}", path.display(), err);
                        }
                    }
                }
            }
        }

        if all_items.is_empty() && file_count == 0 {
            eprintln!(
                "Notice: No data files matching {:?} found in {}",
                self.extensions,
                dir.display()
            );
        }

        Ok(all_items)
    }
}
