use anyhow::{Result, Context};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;

pub struct SelfieSelfSegmentation {
    py_module: PyObject,
    gil_guard: GILGuard,
}

impl SelfieSelfSegmentation {
    pub fn new() -> Result<Self> {
        // Acquire the Python GIL
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        
        // Add the Python module directory to sys.path
        let sys = py.import("sys")?;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("python");
        
        let path_str = path.to_string_lossy().to_string();
        sys.getattr("path")?.call_method1("append", (path_str,))?;
        
        // Import our Python module
        let py_module = PyModule::import(py, "selfie_segmentation_bridge")?
            .to_object(py);
            
        Ok(Self {
            py_module,
            gil_guard,
        })
    }
    
    pub fn process_image(&self, image_data: &[u8]) -> Result<Vec<u8>> {
        let py = self.gil_guard.python();
        
        // Fixed size for the image (256x256)
        let width = 256;
        let height = 256;
        
        // Create a Python bytes object from the image data
        let py_bytes = PyBytes::new(py, image_data);
        
        // Call the Python function
        let result = self.py_module
            .call_method1(
                py,
                "process_image_from_rust",
                (py_bytes, width, height),
            )
            .context("Failed to call Python process_image_from_rust")?;
            
        // Convert the result back to Rust bytes
        let mask_bytes = result.extract::<&PyBytes>(py)?;
        let mask_vec = mask_bytes.as_bytes().to_vec();
        
        Ok(mask_vec)
    }
} 