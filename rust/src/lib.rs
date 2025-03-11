use anyhow::Result;
use godot::classes::image::Format;
use godot::classes::{Image, ImageTexture, ResourceLoader};
use godot::prelude::*;
use image::{Rgb, RgbImage};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType},
    Camera,
};
use std::convert::TryFrom;
use std::sync::Arc;
use std::panic::{self, AssertUnwindSafe};

use mediapipe_rs::tasks::vision::{ImageSegmenterBuilder, ImageSegmenter};
use tokio::runtime::Runtime as TokioRuntime;

mod python_bridge;
use python_bridge::SelfieSelfSegmentation;

struct StreamOverlayArcadeRust;

#[gdextension]
unsafe impl ExtensionLibrary for StreamOverlayArcadeRust {}

#[derive(GodotClass)]
#[class(base = Node)]
struct MediaPipeSegmentation {
    current_frame: Option<Gd<Image>>,
    output_mask: Option<Gd<Image>>,
    segmentation_masks: Vec<Gd<Image>>,
    detection_scores: Vec<f32>,
    detection_classes: Vec<i32>,
    detection_boxes: Vec<Rect2i>,
    runtime: Option<Arc<TokioRuntime>>,
    #[base]
    base: Base<Node>,
    model_loaded: bool,
    detection_threshold: f32,
    input_width: i32,
    input_height: i32,
    class_colors: Vec<Color>,
    segmenter: Option<ImageSegmenter>,
    selfie_segmentation: Option<SelfieSelfSegmentation>,
}

#[godot_api]
impl INode for MediaPipeSegmentation {
    fn init(base: Base<Node>) -> Self {
        let mut instance = Self {
            base,
            current_frame: None,
            output_mask: None,
            segmentation_masks: Vec::new(),
            detection_scores: Vec::new(),
            detection_classes: Vec::new(),
            detection_boxes: Vec::new(),
            runtime: None,
            model_loaded: false,
            detection_threshold: 0.5,
            input_width: 550,
            input_height: 550,
            class_colors: Vec::new(),
            segmenter: None,
            selfie_segmentation: None,
        };
        
        // Try to initialize selfie segmentation
        match SelfieSelfSegmentation::new() {
            Ok(segmentation) => {
                instance.selfie_segmentation = Some(segmentation);
                godot_print!("Selfie segmentation initialized during startup");
            }
            Err(err) => {
                godot_error!("Failed to initialize selfie segmentation during startup: {}", err);
            }
        }
        
        instance
    }
}

#[godot_api]
impl MediaPipeSegmentation {
    #[func]
    fn load_model(&mut self, model_path: String) -> bool {
        // Load the model path as a resource
        let model_path_gstring: GString = model_path.into();
        let mut resource_loader = ResourceLoader::singleton();
        let resource = resource_loader.load(&model_path_gstring);

        let model_resource = match resource {
            Some(res) => res,
            None => {
                godot_error!(
                    "Failed to load model resource from path: {}",
                    model_path_gstring
                );
                return false;
            }
        };

        // Setup class colors
        self.class_colors = vec![];
        for i in 0..256 {
            let hue = (i * 7) % 360;
            let h = hue as f32 / 360.0;
            let s = 0.7;
            let v = 0.9;

            let c = v * s;
            let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
            let m = v - c;

            let (r, g, b) = if h < 1.0 / 6.0 {
                (c, x, 0.0)
            } else if h < 2.0 / 6.0 {
                (x, c, 0.0)
            } else if h < 3.0 / 6.0 {
                (0.0, c, x)
            } else if h < 4.0 / 6.0 {
                (0.0, x, c)
            } else if h < 5.0 / 6.0 {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            };

            let color = Color::from_rgb(r + m, g + m, b + m);
            self.class_colors.push(color);
        }
        self.class_colors.push(Color::from_rgb(1.0, 0.0, 1.0));

        // Get model data as bytes
        let model_data = {
            let data_variant = model_resource.get("data");
            match data_variant.try_to::<PackedByteArray>() {
                Ok(byte_array) => byte_array.to_vec(),
                Err(_) => {
                    godot_error!("Failed to get model data from resource");
                    return false;
                }
            }
        };

        // Create a tokio runtime for async operations
        let tokio_runtime = match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build() {
                Ok(rt) => rt,
                Err(e) => {
                    godot_error!("Failed to create tokio runtime: {}", e);
                    return false;
                }
            };
            
        self.runtime = Some(Arc::new(tokio_runtime));
        
        // Initialize MediaPipe segmenter
        match ImageSegmenterBuilder::new()
            .output_category_mask(true)
            .output_confidence_masks(true)
            .build_from_buffer(&model_data) {
            Ok(segmenter) => {
                self.segmenter = Some(segmenter);
                self.model_loaded = true;
                godot_print!("MediaPipe segmenter loaded successfully");
                true
            },
            Err(e) => {
                godot_error!("Failed to build MediaPipe segmenter: {}", e);
                false
            }
        }
    }

    #[func]
    fn is_ready(&self) -> bool {
        self.model_loaded
    }

    #[func]
    fn set_current_frame(&mut self, image: Gd<Image>) {
        self.current_frame = Some(image);
    }

    #[func]
    fn get_output_mask(&self) -> Gd<Image> {
        match &self.output_mask {
            Some(mask) => mask.clone(),
            None => {
                if let Some(empty_mask) = Image::create(640, 480, false, Format::RGB8) {
                    empty_mask
                } else {
                    godot_error!("Failed to create empty mask");
                    Image::new_gd()
                }
            }
        }
    }

    #[func]
    fn process_all_objects(&mut self) -> bool {
        if !self.is_ready() {
            godot_error!("Model is not ready");
            return false;
        }

        if let Some(image) = &self.current_frame {
            let width = image.get_width();
            let height = image.get_height();

            let mut img_clone = image.clone();
            img_clone.convert(Format::RGB8);
            let image_data = img_clone.get_data();

            match self.segment_with_mediapipe(image_data.as_slice(), width, height) {
                Ok(_) => {
                    return true;
                }
                Err(e) => {
                    godot_error!("Error processing frame: {}", e);
                }
            }
        } else {
            godot_error!("No current frame to process");
        }

        false
    }

    #[func]
    fn set_detection_threshold(&mut self, threshold: f32) -> bool {
        if threshold >= 0.0 && threshold <= 1.0 {
            self.detection_threshold = threshold;
            return true;
        } else {
            godot_error!("Invalid threshold value. Must be between 0.0 and 1.0");
            return false;
        }
    }

    fn segment_with_mediapipe(&mut self, rgb_data: &[u8], width: i32, height: i32) -> Result<()> {
        let expected_size = (width * height * 3) as usize;
        if rgb_data.len() != expected_size {
            godot_error!(
                "Invalid RGB data size: got {}, expected {}",
                rgb_data.len(),
                expected_size
            );
            return Err(anyhow::anyhow!("Invalid RGB data size"));
        }

        // Check if the segmenter is available
        if self.segmenter.is_none() {
            return Err(anyhow::anyhow!("MediaPipe segmenter not initialized"));
        }

        // Convert image data to RGB image
        let rgb_image = self.convert_to_rgb_image(rgb_data, width, height)?;
        
        // Process image with MediaPipe
        let segment_result = self.segmenter.as_mut().unwrap().segment(&rgb_image)?;
        
        // Clear previous detections
        self.detection_scores.clear();
        self.detection_classes.clear();
        self.detection_boxes.clear();
        self.segmentation_masks.clear();
        
        // Process the category mask if available
        if let Some(category_mask) = &segment_result.category_mask {
            // Get dimensions
            let mask_width = i32::try_from(category_mask.width()).unwrap_or(width);
            let mask_height = i32::try_from(category_mask.height()).unwrap_or(height);
            
            // Create an RGB mask image
            let mut mask_image = RgbImage::new(mask_width as u32, mask_height as u32);
            
            // Process each segment in the mask
            for y in 0..mask_height as u32 {
                for x in 0..mask_width as u32 {
                    // Access the pixel directly - get_pixel returns a Luma<u8>
                    let pixel = category_mask.get_pixel(x, y);
                    let category = pixel[0]; // Get the u8 value from the Luma
                    
                    if category > 0 {
                        // Use category value to look up the color
                        let color_idx = category as usize % self.class_colors.len();
                        let color = &self.class_colors[color_idx];
                        
                        let r = (color.r * 255.0) as u8;
                        let g = (color.g * 255.0) as u8;
                        let b = (color.b * 255.0) as u8;
                        
                        mask_image.put_pixel(x, y, Rgb([r, g, b]));
                    } else {
                        // Background
                        mask_image.put_pixel(x, y, Rgb([0, 0, 0]));
                    }
                }
            }
            
            // Create Godot image from mask
            if let Ok(godot_mask) = self.create_single_godot_mask(&mask_image, mask_width, mask_height) {
                self.segmentation_masks.push(godot_mask.clone());
                
                // Set combined mask
                self.output_mask = Some(godot_mask);
                
                // Create detection info
                self.detection_scores.push(1.0); // MediaPipe doesn't provide scores per category
                self.detection_classes.push(1);  // Default class for now
                
                // Create bounding box (full image for now - could be calculated from mask)
                let rect = Rect2i::new(
                    Vector2i::new(0, 0),
                    Vector2i::new(width, height)
                );
                self.detection_boxes.push(rect);
            }
        }
        
        Ok(())
    }

    fn convert_to_rgb_image(&self, rgb_data: &[u8], width: i32, height: i32) -> Result<RgbImage> {
        let mut image = RgbImage::new(width as u32, height as u32);

        for y in 0..height as u32 {
            for x in 0..width as u32 {
                let index = ((y * width as u32 + x) * 3) as usize;
                let r = rgb_data[index];
                let g = rgb_data[index + 1];
                let b = rgb_data[index + 2];
                image.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        Ok(image)
    }

    fn create_godot_mask_from_image(
        &mut self,
        mask: &RgbImage,
        width: i32,
        height: i32,
    ) -> Result<()> {
        let mut godot_mask = match Image::create(width, height, false, Format::RGBA8) {
            Some(img) => img,
            None => {
                godot_error!(
                    "Failed to create Godot image with dimensions {}x{}",
                    width,
                    height
                );
                return Err(anyhow::anyhow!("Failed to create Godot image"));
            }
        };

        let mut byte_array = PackedByteArray::new();
        byte_array.resize((width * height * 4) as usize);

        for y in 0..height as u32 {
            for x in 0..width as u32 {
                if x < mask.width() && y < mask.height() {
                    let pixel = mask.get_pixel(x, y);
                    let index = ((y as i32 * width + x as i32) * 4) as usize;

                    let alpha = if pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0 {
                        128
                    } else {
                        0
                    };

                    byte_array[index] = pixel[0];
                    byte_array[index + 1] = pixel[1];
                    byte_array[index + 2] = pixel[2];
                    byte_array[index + 3] = alpha;
                }
            }
        }

        godot_mask.set_data(width, height, false, Format::RGBA8, &byte_array);
        self.output_mask = Some(godot_mask);

        Ok(())
    }

    fn create_single_godot_mask(
        &self,
        mask: &RgbImage,
        width: i32,
        height: i32,
    ) -> Result<Gd<Image>> {
        let mut godot_mask = match Image::create(width, height, false, Format::RGBA8) {
            Some(img) => img,
            None => {
                return Err(anyhow::anyhow!("Failed to create Godot image"));
            }
        };

        let mut byte_array = PackedByteArray::new();
        byte_array.resize((width * height * 4) as usize);

        for y in 0..height as u32 {
            for x in 0..width as u32 {
                if x < mask.width() && y < mask.height() {
                    let pixel = mask.get_pixel(x, y);
                    let index = ((y as i32 * width + x as i32) * 4) as usize;

                    let alpha = if pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0 {
                        128
                    } else {
                        0
                    };

                    byte_array[index] = pixel[0];
                    byte_array[index + 1] = pixel[1];
                    byte_array[index + 2] = pixel[2];
                    byte_array[index + 3] = alpha;
                }
            }
        }

        godot_mask.set_data(width, height, false, Format::RGBA8, &byte_array);

        Ok(godot_mask)
    }

    #[func]
    fn get_segmentation_masks(&self) -> VariantArray {
        let mut array = VariantArray::new();
        for mask in &self.segmentation_masks {
            array.push(&mask.clone().to_variant());
        }
        array
    }

    #[func]
    fn get_combined_mask(&self) -> Gd<Image> {
        self.get_output_mask()
    }

    #[func]
    fn get_detection_scores(&self) -> PackedFloat32Array {
        PackedFloat32Array::from(self.detection_scores.clone())
    }

    #[func]
    fn get_detection_classes(&self) -> PackedInt32Array {
        PackedInt32Array::from(self.detection_classes.clone())
    }

    #[func]
    fn get_detection_boxes(&self) -> Array<Rect2i> {
        let mut array = Array::new();
        for rect in &self.detection_boxes {
            array.push(*rect);
        }
        array
    }

    #[func]
    fn get_class_color(&self, class_id: i32) -> Color {
        if class_id < 0 || class_id as usize >= self.class_colors.len() {
            return Color::from_rgb(1.0, 0.0, 1.0);
        }
        self.class_colors[class_id as usize]
    }

    #[func]
    fn get_detection_count(&self) -> i32 {
        self.detection_scores.len() as i32
    }

    #[func]
    fn get_detection_at_index(&self, index: i32) -> Dictionary {
        let mut dict = Dictionary::new();

        if index < 0 || index as usize >= self.detection_scores.len() {
            return dict;
        }

        let idx = index as usize;
        dict.set("score", self.detection_scores[idx]);
        dict.set("class_id", self.detection_classes[idx]);

        if idx < self.detection_boxes.len() {
            let rect = self.detection_boxes[idx];
            dict.set("x", rect.position.x);
            dict.set("y", rect.position.y);
            dict.set("width", rect.size.x);
            dict.set("height", rect.size.y);
        }

        if idx < self.segmentation_masks.len() {
            dict.set("mask", self.segmentation_masks[idx].clone());
        }

        dict
    }

    #[func]
    fn init_selfie_segmentation(&mut self) -> bool {
        match SelfieSelfSegmentation::new() {
            Ok(segmentation) => {
                self.selfie_segmentation = Some(segmentation);
                godot_print!("Selfie segmentation initialized successfully");
                true
            }
            Err(err) => {
                godot_error!("Failed to initialize selfie segmentation: {}", err);
                false
            }
        }
    }

    #[func]
    fn process_image_with_python(&mut self, image: Gd<Image>) -> Gd<Image> {
        let selfie_segmentation = match &self.selfie_segmentation {
            Some(segmentation) => segmentation,
            None => {
                godot_error!("Selfie segmentation not initialized");
                return Image::new().upcast();
            }
        };

        // Get image data
        let mut image_ref = image.bind();
        image_ref.convert(Format::FORMAT_RGB8);
        
        // Resize to 256x256 if needed
        if image_ref.get_width() != 256 || image_ref.get_height() != 256 {
            image_ref.resize(256, 256, Image::INTERPOLATE_BILINEAR);
        }
        
        let image_data = image_ref.get_data();
        
        // Process the image
        let mask_data = match selfie_segmentation.process_image(&image_data) {
            Ok(data) => data,
            Err(err) => {
                godot_error!("Failed to process image: {}", err);
                return Image::new().upcast();
            }
        };
        
        // Create a new image from the mask data
        let mut mask_image = Image::new();
        mask_image.set_data(256, 256, false, Format::FORMAT_L8, PackedByteArray::from(mask_data));
        
        mask_image.upcast()
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
