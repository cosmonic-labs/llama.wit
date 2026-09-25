//! Hugging Face models, cached in the preopened directory so they download once.

use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use wstd::http::{Body, BodyExt, Client, Method, Request, Uri};

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

pub fn load(model: &HfModel) -> Result<Vec<u8>, String> {
    download(model).map_err(|e| format!("{}: {e}", model.file))
}

fn download(model: &HfModel) -> Result<Vec<u8>, Box<dyn Error>> {
    let path = PathBuf::from(CACHE_DIR).join(model.file);
    if !path.exists() {
        fs::create_dir_all(CACHE_DIR)?;
        let partial = path.with_extension("part");
        wstd::runtime::block_on(fetch(model.url, &partial))?;
        fs::rename(&partial, &path)?;
    }
    Ok(fs::read(&path)?)
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
