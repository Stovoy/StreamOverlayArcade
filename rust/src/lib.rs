use godot::classes::image::Format;
use godot::classes::{Image, ImageTexture};
use godot::prelude::*;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType},
    Camera,
};

use ndarray::Array3;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::panic::AssertUnwindSafe;
use std::panic;

struct StreamOverlayArcadeRust;

#[gdextension]
unsafe impl ExtensionLibrary for StreamOverlayArcadeRust {
    fn on_level_init(level: InitLevel) {
        if level == InitLevel::Scene {
            godot_print!("Initializing Python interpreter for StreamOverlayArcade extension");
            let _ = pyo3::prepare_freethreaded_python();
        }
    }
}

#[derive(GodotClass)]
#[class(base = Node)]
struct MediaPipeSegmentation {
    current_frame: Option<Gd<Image>>,
    #[base]
    base: Base<Node>,
    py_segmentation: Option<Py<PyAny>>,
}

#[godot_api]
impl INode for MediaPipeSegmentation {
    fn init(base: Base<Node>) -> Self {
        let _ = pyo3::prepare_freethreaded_python();
        
        let py_segmentation = Python::with_gil(|py| {
            match py.import_bound("python.segmentation") {
                Ok(module) => {
                    godot_print!("Segmentation initialized successfully");
                    Some(module.into_py(py))
                }
                Err(err) => {
                    godot_error!("Failed to initialize segmentation: {}", err);
                    None
                }
            }
        });
        
        Self {
            base,
            current_frame: None,
            py_segmentation,
        }
    }
    
    fn ready(&mut self) {
        godot_print!("MediaPipeSegmentation is ready");
    }
}

#[godot_api]
impl MediaPipeSegmentation {
    #[func]
    fn is_ready(&self) -> bool {
        self.py_segmentation.is_some()
    }

    #[func]
    fn set_current_frame(&mut self, image: Gd<Image>) {
        self.current_frame = Some(image);
    }

    #[func]
    fn process_segmentation(&self) -> Dictionary {
        let mut result_dict = Dictionary::new();
        
        if self.py_segmentation.is_none() {
            godot_error!("Segmentation not initialized");
            return result_dict;
        }
        
        if let Some(image) = &self.current_frame {
            // Create a copy of the image and convert to RGB8
            let mut image_copy = image.clone();
            image_copy.convert(Format::RGB8);
            
            // Get dimensions and data
            let width = image_copy.get_width() as usize;
            let height = image_copy.get_height() as usize;
            let image_data = image_copy.get_data();
            
            // Convert the image data into ndarray format
            let mut array = Array3::<u8>::zeros((height, width, 3));
            let mut idx = 0;
            
            for y in 0..height {
                for x in 0..width {
                    if idx + 2 < image_data.len() {
                        array[[y, x, 0]] = image_data[idx];
                        array[[y, x, 1]] = image_data[idx + 1];
                        array[[y, x, 2]] = image_data[idx + 2];
                    }
                    idx += 3;
                }
            }
            
            // Process the image with Python
            let segmentation_result = Python::with_gil(|py| {
                let segmentation = self.py_segmentation.as_ref().unwrap().bind(py);
                
                // Convert Rust array to numpy array
                let image_array = numpy::PyArray::from_array_bound(py, &array);
                
                // Call the process_image function
                match segmentation.call_method1("process_image", (image_array,)) {
                    Ok(result) => {
                        // Clone the result to extend its lifetime
                        Some(result.into_py(py))
                    },
                    Err(err) => {
                        godot_error!("Failed to call process_image: {}", err);
                        None
                    }
                }
            });
            
            if let Some(py_result) = segmentation_result {
                Python::with_gil(|py| {
                    // Extract results from Python
                    let py_result = py_result.bind(py);
                    if let Ok(tuple) = py_result.downcast::<PyTuple>() {
                        if tuple.len() >= 4 {
                            // Get the colored mask
                            if let Ok(colored_mask_obj) = tuple.get_item(0) {
                                if let Ok(colored_mask) = colored_mask_obj.extract::<&numpy::PyArray3<u8>>() {
                                    let colored_mask_array = colored_mask.to_owned_array();
                                    let mask_height = colored_mask_array.shape()[0];
                                    let mask_width = colored_mask_array.shape()[1];
                                    
                                    // Create mask image
                                    let mut mask_bytes = Vec::with_capacity(mask_height * mask_width * 4);
                                    for y in 0..mask_height {
                                        for x in 0..mask_width {
                                            mask_bytes.push(colored_mask_array[[y, x, 0]]);
                                            mask_bytes.push(colored_mask_array[[y, x, 1]]);
                                            mask_bytes.push(colored_mask_array[[y, x, 2]]);
                                            
                                            let alpha = if colored_mask_array[[y, x, 0]] > 0 || 
                                                          colored_mask_array[[y, x, 1]] > 0 || 
                                                          colored_mask_array[[y, x, 2]] > 0 {
                                                128
                                            } else {
                                                0
                                            };
                                            mask_bytes.push(alpha);
                                        }
                                    }
                                    
                                    let mut mask_image = Image::new_gd();
                                    let byte_array = PackedByteArray::from(mask_bytes);
                                    mask_image.set_data(
                                        mask_width as i32, 
                                        mask_height as i32, 
                                        false,
                                        Format::RGBA8, 
                                        &byte_array
                                    );
                                    
                                    result_dict.set("mask", mask_image);
                                }
                            }
                            
                            // Get category info
                            if let Ok(category_info_obj) = tuple.get_item(3) {
                                if let Ok(category_info) = category_info_obj.downcast::<PyList>() {
                                    let mut detections = VariantArray::new();
                                    
                                    for i in 0..category_info.len() {
                                        if let Ok(cat) = category_info.get_item(i) {
                                            if let Ok(cat_dict) = cat.downcast::<PyDict>() {
                                                let mut detection = Dictionary::new();
                                                
                                                if let (Ok(Some(id_item)), Ok(Some(name_item)), Ok(Some(color_item))) = (
                                                    cat_dict.get_item("id"),
                                                    cat_dict.get_item("name"),
                                                    cat_dict.get_item("color")
                                                ) {
                                                    if let Ok(id) = id_item.extract::<u32>() {
                                                        detection.set("class_id", id as i32);
                                                        
                                                        if let Ok(name) = name_item.extract::<String>() {
                                                            detection.set("class_name", name);
                                                        }
                                                        
                                                        if let Ok(color_tuple) = color_item.downcast::<PyTuple>() {
                                                            if color_tuple.len() == 3 {
                                                                if let (Ok(r), Ok(g), Ok(b)) = (
                                                                    color_tuple.get_item(0).and_then(|v| v.extract::<u8>()),
                                                                    color_tuple.get_item(1).and_then(|v| v.extract::<u8>()),
                                                                    color_tuple.get_item(2).and_then(|v| v.extract::<u8>())
                                                                ) {
                                                                    detection.set("color", Color::from_rgb(
                                                                        r as f32 / 255.0,
                                                                        g as f32 / 255.0,
                                                                        b as f32 / 255.0
                                                                    ));
                                                                }
                                                            }
                                                        }
                                                        
                                                        detection.set("score", 1.0);
                                                        detections.push(&detection.to_variant());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                    result_dict.set("detections", detections);
                                }
                            }
                        }
                    }
                });
            }
        } else {
            godot_error!("No current frame to process");
        }

        result_dict
    }
}



#[derive(GodotClass)]
#[class(base = Node)]
pub struct WebCameraManager {
    #[base]
    base: Base<Node>,
    camera: Option<Camera>,
    image: Option<Gd<Image>>,
    texture: Option<Gd<ImageTexture>>,
}

#[godot_api]
impl INode for WebCameraManager {
    fn init(base: Base<Node>) -> Self {
        Self {
            base,
            camera: None,
            image: None,
            texture: None,
        }
    }

    fn ready(&mut self) {
        if let Some(image) = Image::create(640, 480, false, Format::RGB8) {
            self.image = Some(image);
            if let Some(img_ref) = &self.image {
                if let Some(texture) = ImageTexture::create_from_image(img_ref) {
                    self.texture = Some(texture);
                } else {
                    godot_print!("Error: Failed to create texture from initial image");
                }
            }
        } else {
            godot_print!("Error: Failed to create initial image");
        }
    }
}

#[godot_api]
impl WebCameraManager {
    #[func]
    fn start_camera(&mut self) -> bool {
        if self.camera.is_some() {
            self.stop_camera();
        }

        let candidate_resolutions = vec![
            (3840, 2160),
            (2560, 1440),
            (1920, 1080),
            (1600, 1200),
            (1280, 720),
            (1024, 768),
            (800, 600),
            (640, 480),
            (320, 240),
        ];
        let candidate_fps = vec![60, 30, 15];
        let candidate_formats = vec![FrameFormat::MJPEG, FrameFormat::YUYV];

        let camera_index = CameraIndex::Index(0);

        for (width, height) in candidate_resolutions {
            for &fps in &candidate_fps {
                for &frame_format in &candidate_formats {
                    let requested_format =
                        RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
                            CameraFormat::new_from(width, height, frame_format, fps),
                        ));

                    let camera_result = Camera::new(camera_index.clone(), requested_format);

                    let mut camera = match camera_result {
                        Ok(cam) => cam,
                        Err(_) => {
                            continue;
                        }
                    };

                    match camera.open_stream() {
                        Ok(_) => {
                            let camera_format = camera.camera_format();
                            godot_print!(
                                "Camera opened successfully with format: {:?}",
                                camera_format
                            );

                            let width = match i32::try_from(camera_format.width()) {
                                Ok(w) => w,
                                Err(_) => {
                                    godot_error!("Invalid camera width");
                                    continue;
                                }
                            };

                            let height = match i32::try_from(camera_format.height()) {
                                Ok(h) => h,
                                Err(_) => {
                                    godot_error!("Invalid camera height");
                                    continue;
                                }
                            };

                            if width <= 0 || height <= 0 {
                                godot_error!("Invalid camera dimensions: {}x{}", width, height);
                                continue;
                            }

                            let image = match Image::create(width, height, false, Format::RGB8) {
                                Some(img) => img,
                                None => {
                                    godot_error!(
                                        "Failed to create image with dimensions {}x{}",
                                        width,
                                        height
                                    );
                                    continue;
                                }
                            };

                            let texture = match ImageTexture::create_from_image(&image) {
                                Some(tex) => tex,
                                None => {
                                    godot_error!(
                                        "Failed to create texture from image with dimensions {}x{}",
                                        width,
                                        height
                                    );
                                    continue;
                                }
                            };

                            self.image = Some(image);
                            self.texture = Some(texture);
                            self.camera = Some(camera);

                            godot_print!(
                                "Camera started successfully with dimensions {}x{}",
                                width,
                                height
                            );
                            return true;
                        }
                        Err(e) => {
                            godot_print!(
                                "Failed to open stream with format {}x{} @ {}fps: {:?}",
                                width,
                                height,
                                fps,
                                e
                            );

                            continue;
                        }
                    }
                }
            }
        }

        godot_error!("All camera format attempts failed");
        false
    }

    #[func]
    fn capture_frame(&mut self) -> bool {       
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            if let Some(camera) = &mut self.camera {
                match camera.frame() {
                    Ok(frame) => match frame.decode_image::<RgbFormat>() {
                        Ok(rgb_frame) => {
                            let rgb_data = rgb_frame.to_vec();
                            let width = i32::try_from(rgb_frame.width()).unwrap_or(0);
                            let height = i32::try_from(rgb_frame.height()).unwrap_or(0);
                            if let Some(image) = &mut self.image {
                                let byte_array = PackedByteArray::from(rgb_data);
                                image.set_data(width, height, false, Format::RGB8, &byte_array);
                                if let Some(texture) = &mut self.texture {
                                    texture.update(&*image);
                                    return true;
                                } else {
                                    godot_print!("Error: Texture is None");
                                }
                            } else {
                                godot_print!("Error: Image is None");
                            }
                        }
                        Err(err) => {
                            godot_print!("Error decoding image: {}", err);
                        }
                    },
                    Err(err) => {
                        godot_print!("Error capturing frame: {}", err);
                    }
                }
            }
            false
        }));
        
        match result {
            Ok(success) => success,
            Err(e) => {
                if let Some(error_msg) = e.downcast_ref::<String>() {
                    godot_print!("Panic in capture_frame: {}", error_msg);
                } else if let Some(error_msg) = e.downcast_ref::<&str>() {
                    godot_print!("Panic in capture_frame: {}", error_msg);
                } else {
                    godot_print!("Unknown panic in capture_frame");
                }
                false
            }
        }
    }

    #[func]
    fn stop_camera(&mut self) {
        if self.camera.is_some() {
            self.camera = None;
            godot_print!("Camera stopped");
        }
    }

    #[func]
    fn get_texture(&self) -> Option<Gd<ImageTexture>> {
        self.texture.clone()
    }

    #[func]
    fn get_image(&self) -> Option<Gd<Image>> {
        self.image.clone()
    }
}
