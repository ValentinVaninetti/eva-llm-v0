use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/gpu/shaders");
    let src = PathBuf::from("src/gpu/shaders");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::create_dir_all(&out).unwrap();

    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("comp") {
            let name = entry.file_name().to_string_lossy().to_string();
            let stem = name.trim_end_matches(".comp");
            let spv = out.join(format!("{}.spv", stem));
            let status = Command::new("glslc")
                .arg("-c")
                .arg("-O")
                .arg("-o")
                .arg(&spv)
                .arg(&path)
                .status()
                .expect("glslc is not installed (install glslang-tools or shaderc). Without glslc the Vulkan shaders cannot be compiled.");
            assert!(status.success(), "glslc failed to compile {}", name);
        }
    }
}
