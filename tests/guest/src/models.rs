//! Hugging Face models, cached in the preopened directory so they download once.
//!
//! Models are never read into memory: GGUF files run to tens of gigabytes,
//! far past what a wasm32 guest can address. `stream` hands the provider a
//! `stream<u8>` fed chunk-by-chunk off the file instead.

use std::cell::RefCell;
use std::error::Error;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use wit_bindgen::StreamReader;
use wstd::http::{Body, BodyExt, Client, Method, Request, Uri};

use crate::wit_stream;

const CACHE_DIR: &str = "/models";

/// How much of the file to hand the provider per write. Bounded so that
/// neither side has to hold the whole model, and well under the Canonical
/// ABI's per-read cap of `2**28 - 1` items.
const CHUNK: usize = 4 << 20; // 4 MiB

pub struct HfModel {
    /// `None` for models too big to pull down during a test run. Put the file
    /// in `tests/guest-dir/models/` by hand and the test picks it up; without
    /// it the test skips.
    url: Option<&'static str>,
    file: &'static str,
}

/// ~1.1 MiB
pub const STORIES_260K: HfModel = HfModel {
    url: Some("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"),
    file: "stories260K.gguf",
};

/// Live suite. Downloaded on first use.
pub const QWEN: HfModel = HfModel {
    url: Some("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    file: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
};

pub const LLAMA: HfModel = HfModel {
    url: Some("https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
};

pub const BGE: HfModel = HfModel {
    url: Some("https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf"),
    file: "bge-small-en-v1.5-q8_0.gguf",
};

/// ~4.7 GiB, Q4_K_M. Past the 4 GiB mark where a model stops fitting in the
/// provider's address space, but small enough to sit well inside a typical
/// GPU's memory -- the size where streaming has to work and memory pressure is
/// not yet a factor, which is what separates the two when a bigger model fails.
pub const QWEN3_8B: HfModel = HfModel {
    url: Some("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"),
    file: "Qwen3-8B-Q4_K_M.gguf",
};

/// ~8.3 GiB, Q2_K. A bisection point between the largest model known to load
/// intact and the smallest known to fail, at the same architecture.
/// `llama-quantize --allow-requantize <Q8_0> <Q2_K> Q2_K`
pub const MISTRAL_SMALL_24B_Q2: HfModel = HfModel {
    url: None,
    file: "Mistral-Small-24B-Instruct-2501-Q2_K.gguf",
};

/// ~9.7 GiB, Q3_K_S. A second bisection point.
/// `llama-quantize --allow-requantize <Q8_0> <Q3_K_S> Q3_K_S`
pub const MISTRAL_SMALL_24B_Q3: HfModel = HfModel {
    url: None,
    file: "Mistral-Small-24B-Instruct-2501-Q3_K_S.gguf",
};

/// ~10.7 GiB, Q3_K_M. A third bisection point.
pub const MISTRAL_SMALL_24B_Q3M: HfModel = HfModel {
    url: None,
    file: "Mistral-Small-24B-Instruct-2501-Q3_K_M.gguf",
};

/// ~13.3 GiB, Q4_K_M. The same 24B as below at half the weight, which is what
/// fits on a 32 GiB machine -- and a point midway through the range where
/// loading goes from working to not.
///
/// Not downloaded: requantize the Q8_0 in place, which needs no network:
/// `llama-quantize --allow-requantize <Q8_0> <Q4_K_M> Q4_K_M`
pub const MISTRAL_SMALL_24B_Q4: HfModel = HfModel {
    url: None,
    file: "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
};

/// ~23 GiB, Q8_0. Needs a GPU backend: at this size the weights only fit
/// because each chunk is uploaded and dropped instead of kept in the
/// provider's linear memory.
pub const MISTRAL_SMALL_24B: HfModel = HfModel {
    url: None,
    file: "Mistral-Small-24B-Instruct-2501-Q8_0.gguf",
};

impl HfModel {
    pub fn name(&self) -> &'static str {
        self.file
    }

    fn path(&self) -> PathBuf {
        PathBuf::from(CACHE_DIR).join(self.file)
    }

    /// Whether the file is already on disk, without downloading anything.
    pub fn is_cached(&self) -> bool {
        self.path().exists()
    }
}

/// Feed `model` into a `stream<u8>` for `Model::create`, along with its total
/// size (the loader needs the length up front; streams don't carry one).
///
/// The writer runs as its own task because the provider pulls bytes off the
/// stream *during* `create` — writing inline would deadlock against it.
pub fn stream(model: &HfModel) -> Result<(StreamReader<u8>, u64, ReadErrors), String> {
    let path = ensure(model)?;
    let open = || -> Result<(File, u64), std::io::Error> {
        let file = File::open(&path)?;
        let size = file.metadata()?.len();
        Ok((file, size))
    };
    let (mut file, size) = open().map_err(|e| format!("{}: {e}", model.file))?;

    let (mut tx, rx) = wit_stream::new::<u8>();
    let errors = ReadErrors::default();
    let sink = errors.clone();
    wit_bindgen::spawn_local(async move {
        let mut buf = vec![0u8; CHUNK];
        loop {
            let n = match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    // Nothing to return to here: drop the writer, which ends
                    // the stream short and fails `create`. `explain` pairs
                    // that failure back up with this error.
                    sink.set(format!("read failed: {e}"));
                    break;
                }
            };
            if !tx.write_all(buf[..n].to_vec()).await.is_empty() {
                break; // provider hung up, it reports its own error
            }
        }
    });

    Ok((rx, size, errors))
}

/// Carries a streaming failure out of the writer task, which has no way to
/// report one itself.
#[derive(Default, Clone)]
pub struct ReadErrors(Rc<RefCell<Option<String>>>);

impl ReadErrors {
    fn set(&self, error: String) {
        *self.0.borrow_mut() = Some(error);
    }

    /// Fold a stream-side failure into a `create` error, so a bad disk read
    /// doesn't surface as an unexplained truncated-model error.
    pub fn explain(&self, error: String) -> String {
        match self.0.borrow().as_ref() {
            Some(cause) => format!("{error} (streaming the file failed: {cause})"),
            None => error,
        }
    }
}

/// Path to the model's file, downloading it first if it isn't cached yet.
fn ensure(model: &HfModel) -> Result<PathBuf, String> {
    let path = model.path();
    if path.exists() {
        return Ok(path);
    }
    let Some(url) = model.url else {
        return Err(format!(
            "{} is not in {CACHE_DIR} and is too large to download automatically",
            model.file
        ));
    };
    download(url, &path).map_err(|e| format!("{}: {e}", model.file))?;
    Ok(path)
}

fn download(url: &str, path: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(CACHE_DIR)?;
    let partial = path.with_extension("part");
    wstd::runtime::block_on(fetch(url, &partial))?;
    fs::rename(&partial, path)?;
    Ok(())
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
