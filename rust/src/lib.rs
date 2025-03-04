use godot::prelude::*;
use godot::classes::{Image, ImageTexture};
use godot::classes::image::Format;
use nokhwa::{
    Camera,
    utils::{CameraFormat, FrameFormat, CameraIndex, RequestedFormat, RequestedFormatType},
    pixel_format::RgbFormat,
};
use std::convert::TryFrom;

struct StreamSilhouetteRust;

#[gdextension]
unsafe impl ExtensionLibrary for StreamSilhouetteRust {}

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
        let mut selected_camera = None;
        let mut selected_camera_format = None;
        'outer: for (width, height) in candidate_resolutions {
            for &fps in &candidate_fps {
                for &frame_format in &candidate_formats {
                    let requested_format = RequestedFormat::new::<RgbFormat>(
                        RequestedFormatType::Closest(CameraFormat::new_from(width, height, frame_format, fps))
                    );
                    if let Ok(mut camera) = Camera::new(CameraIndex::Index(0), requested_format) {
                        if camera.open_stream().is_ok() {
                            selected_camera_format = Some(camera.camera_format());
                            selected_camera = Some(camera);
                            break 'outer;
                        }
                    }
                }
            }
        }
        if let Some(camera) = selected_camera {
            if let Some(actual_format) = selected_camera_format {
                let width = i32::try_from(actual_format.width()).unwrap_or(640);
                let height = i32::try_from(actual_format.height()).unwrap_or(480);
                if let Some(image) = Image::create(width, height, false, Format::RGB8) {
                    self.image = Some(image);
                    if let Some(img_ref) = &self.image {
                        if let Some(texture) = ImageTexture::create_from_image(img_ref) {
                            self.texture = Some(texture);
                        } else {
                            godot_print!("Error: Failed to create texture from image with dimensions {}x{}", width, height);
                            return false;
                        }
                    }
                } else {
                    godot_print!("Error: Failed to create image with dimensions {}x{}", width, height);
                    return false;
                }
            }
            self.camera = Some(camera);
            return true;
        }
        godot_print!("Error: All camera format attempts failed");
        false
    }
    
    #[func]
    fn capture_frame(&mut self) -> bool {
        if let Some(camera) = &mut self.camera {
            match camera.frame() {
                Ok(frame) => {
                    match frame.decode_image::<RgbFormat>() {
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
                        },
                        Err(err) => {
                            godot_print!("Error: Failed to decode RGB frame: {:?}", err);
                        }
                    }
                },
                Err(err) => {
                    godot_print!("Error: Failed to get frame: {:?}", err);
                }
            }
        } else {
            godot_print!("Error: Camera is None");
        }
        false
    }

    #[func]
    fn stop_camera(&mut self) {
        self.camera = None;
    }

    #[func]
    fn get_texture(&self) -> Option<Gd<ImageTexture>> {
        self.texture.clone()
    }
}
