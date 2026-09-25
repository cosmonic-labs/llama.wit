use clap::Parser;
use colored::Colorize;
use std::sync::Arc;
use wasi_webgpu_wasmtime::{
    WasiWebGpuCtx, WasiWebGpuCtxView,
    reexports::{wgpu_core, wgpu_types},
};
use wasmtime::{
    Config, Engine, Store,
    component::{Component, Linker},
    error::Context as _,
};
use wasmtime_wasi::{ResourceTable, WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};
use wasmtime_wasi_http::{
    WasiHttpCtx,
    p2::{WasiHttpCtxView as P2WasiHttpCtxView, WasiHttpView as P2WasiHttpView},
    p3::{WasiHttpCtxView, WasiHttpView},
};

#[derive(clap::Parser, Debug)]
struct RuntimeArgs {
    /// Path to the component
    #[arg(long, short)]
    path: String,
}

wasmtime::component::bindgen!({
    path: "../../wit/",
    world: "test-harness",
});

struct HostState {
    instance: Arc<wgpu_core::global::Global>,
}

impl HostState {
    fn new() -> Self {
        Self {
            instance: Arc::new(wgpu_core::global::Global::new(
                "WebGPU",
                wgpu_types::InstanceDescriptor {
                    backends: wgpu_types::Backends::all(),
                    flags: wgpu_types::InstanceFlags::from_build_config(),
                    backend_options: Default::default(),
                    memory_budget_thresholds: Default::default(),
                    display: None,
                },
                None,
            )),
        }
    }

    pub fn add_workload(&self) -> WorkloadState {
        WorkloadState {
            table: ResourceTable::new(),
            wasi_ctx: WasiCtxBuilder::new()
                .inherit_stdio()
                .inherit_env()
                .preopened_dir(
                    "./guest-dir",
                    "/",
                    wasmtime_wasi::DirPerms::all(),
                    wasmtime_wasi::FilePerms::all(),
                )
                .expect("Failed to preopen dir")
                .build(),
            wasi_http_ctx: WasiHttpCtx::default(),
            instance: Arc::clone(&self.instance),
        }
    }
}

struct WorkloadState {
    table: ResourceTable,
    wasi_ctx: WasiCtx,
    wasi_http_ctx: WasiHttpCtx,
    instance: Arc<wgpu_core::global::Global>,
}

impl wasmtime::component::HasData for WorkloadState {
    type Data<'a> = &'a mut WorkloadState;
}

impl WasiView for WorkloadState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi_ctx,
            table: &mut self.table,
        }
    }
}

impl WasiHttpView for WorkloadState {
    fn http(&mut self) -> WasiHttpCtxView<'_> {
        WasiHttpCtxView {
            ctx: &mut self.wasi_http_ctx,
            table: &mut self.table,
            hooks: Default::default(),
        }
    }
}

impl P2WasiHttpView for WorkloadState {
    fn http(&mut self) -> P2WasiHttpCtxView<'_> {
        P2WasiHttpCtxView {
            ctx: &mut self.wasi_http_ctx,
            table: &mut self.table,
            hooks: Default::default(),
        }
    }
}

impl WasiWebGpuCtxView for WorkloadState {
    fn webgpu_ctx(&mut self) -> WasiWebGpuCtx<'_> {
        WasiWebGpuCtx {
            instance: &self.instance,
            table: &mut self.table,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = RuntimeArgs::parse();

    let host_state = HostState::new();

    let mut config = Config::default();
    config.wasm_component_model(true);
    config.wasm_component_model_async(true);
    let engine = Engine::new(&config)?;

    let mut linker: Linker<WorkloadState> = Linker::new(&engine);
    // p2 for the Rust and p3 for the C
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
    wasmtime_wasi_http::p2::add_only_http_to_linker_async(&mut linker)?;
    wasmtime_wasi::p3::add_to_linker(&mut linker)?;
    wasmtime_wasi_http::p3::add_to_linker(&mut linker)?;
    wasi_webgpu_wasmtime::add_to_linker(&mut linker)?;

    let mut store = Store::new(&engine, host_state.add_workload());

    let component = Component::from_file(&engine, &args.path)
        .with_context(|| format!("failed to load component {}", args.path))?;

    let component = TestHarness::instantiate_async(&mut store, &component, &linker).await?;

    let tests = component
        .func_list_tests()
        .call_async(&mut store, ())
        .await?
        .0;
    println!("running {} tests: {}", tests.len(), tests.join(", "));

    let mut success = 0u32;
    let mut fail = 0u32;

    for test in &tests {
        let result = component
            .func_run_test()
            .call_async(&mut store, (test.to_owned(),))
            .await;
        match result {
            Ok(_) => {
                success += 1;
            }
            Err(e) => {
                fail += 1;
                eprintln!("{e}");
            }
        }
    }

    let message = format!(
        "Ran {test_count} tests, {success} succeeded and {fail} failed",
        test_count = tests.len()
    );
    if fail > 0 {
        let message = message.red();
        anyhow::bail!("{message}");
    } else {
        let message = message.green();
        println!("{message}");
    }

    Ok(())
}
