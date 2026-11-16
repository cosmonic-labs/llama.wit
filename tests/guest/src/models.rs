//! Hugging Face models, cached in the preopened directory so they download once.

use std::cell::RefCell;
use std::error::Error;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use wit_bindgen::StreamReader;
use wstd::http::{Body, BodyExt, Client, Method, Request, Uri};

use crate::wit_stream;

const CACHE_DIR: &str = "/models";

pub struct HfModel {
    url: &'static str,
    file: &'static str,
}

/// ~1.1 MiB
pub const STORIES_260K: HfModel = HfModel {
    url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf",
    file: "stories260K.gguf",
};

pub const QWEN: HfModel = HfModel {
    url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    file: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
};

pub const LLAMA: HfModel = HfModel {
    url: "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
};

pub const BGE: HfModel = HfModel {
    url: "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
    file: "bge-small-en-v1.5-q8_0.gguf",
};

/// A model as a `stream<u8>` for `Model::create`.
pub struct ModelStream {
    pub data: StreamReader<u8>,
    pub size: u64,
    pub read_error: ReadError,
}

/// Set if reading the cached file failed partway. The provider then only sees
/// a short stream, so this says why.
#[derive(Default)]
pub struct ReadError(Rc<RefCell<Option<String>>>);

impl ReadError {
    pub fn is_set(&self) -> bool {
        self.0.borrow().is_some()
    }

    /// Adds the read error, if any, to `create`'s error.
    pub fn explain(&self, create_error: String) -> String {
        match &*self.0.borrow() {
            Some(io) => format!("{create_error} ({io})"),
            None => create_error,
        }
    }
}

/// Streams the cached file a chunk at a time, so the guest never holds it whole
/// and the provider sees partial and blocked reads.
pub fn load(model: &HfModel) -> Result<ModelStream, String> {
    let path = download(model).map_err(|e| file_err(model, e))?;
    let mut file = File::open(&path).map_err(|e| file_err(model, e))?;
    let size = file.metadata().map_err(|e| file_err(model, e))?.len();
    let read_error = ReadError::default();
    let task_error = read_error.0.clone();
    let name = model.file;
    let (mut tx, data) = wit_stream::new::<u8>();
    // A separate task, since the provider reads the stream during `create`
    wit_bindgen::spawn_local(async move {
        // One write must stay under the Canonical ABI's 2^28 - 1 item limit
        const CHUNK: u64 = 4 << 20;
        // `write_all` hands the buffer back, so one allocation serves every chunk
        let mut chunk = Vec::with_capacity(CHUNK.min(size) as usize);
        loop {
            chunk.clear();
            match (&mut file).take(CHUNK).read_to_end(&mut chunk) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => {
                    *task_error.borrow_mut() = Some(format!("{name}: {e}"));
                    break;
                }
            }
            chunk = tx.write_all(chunk).await;
            if !chunk.is_empty() {
                break; // the provider dropped the stream
            }
        }
    });
    Ok(ModelStream {
        data,
        size,
        read_error,
    })
}

/// The whole cached file, for tests that build their own stream.
pub fn read(model: &HfModel) -> Result<Vec<u8>, String> {
    let path = download(model).map_err(|e| file_err(model, e))?;
    fs::read(&path).map_err(|e| file_err(model, e))
}

fn file_err(model: &HfModel, e: impl Display) -> String {
    format!("{}: {e}", model.file)
}

/// The cached file's path, downloading it first if needed.
fn download(model: &HfModel) -> Result<PathBuf, Box<dyn Error>> {
    let path = PathBuf::from(CACHE_DIR).join(model.file);
    if !path.exists() {
        fs::create_dir_all(CACHE_DIR)?;
        let partial = path.with_extension("part");
        wstd::runtime::block_on(fetch(model.url, &partial))?;
        fs::rename(&partial, &path)?;
    }
    Ok(path)
}

async fn fetch(url: &str, dest: &Path) -> Result<(), Box<dyn Error>> {
    let mut url: Uri = url.parse()?;

    // `wstd` does not follow redirects automatically
    const MAX_REDIRECTS: u8 = 8;
    for _ in 0..MAX_REDIRECTS {
        let request = Request::builder()
            .uri(url.clone())
            .method(Method::GET)
            .body(Body::empty())?;
        let response = Client::new().send(request).await?;
        let status = response.status();

        if status.is_redirection() {
            let location = response
                .headers()
                .get("location")
                .ok_or_else(|| format!("{url}: HTTP {status} with no location"))?
                .to_str()?
                .to_string();
            // make location url absolute
            url = if location.starts_with('/') {
                let mut parts = url.into_parts();
                parts.path_and_query = Some(location.parse()?);
                Uri::from_parts(parts)?
            } else {
                location.parse()?
            };
            continue;
        }
        if !status.is_success() {
            return Err(format!("{url}: HTTP {status}").into());
        }

        let mut file = File::create(dest)?;
        let mut body = response.into_body().into_boxed_body();
        while let Some(frame) = body.frame().await {
            if let Some(data) = frame?.data_ref() {
                file.write_all(data)?;
            }
        }
        return Ok(());
    }

    Err("too many redirects".into())
}
