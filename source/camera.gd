extends Node2D

@onready var web_camera_manager = $WebCameraManager
@onready var camera_display = $CameraDisplay
@onready var mask_display = $MaskDisplay
@onready var segmentation = $YolactSegmentation

var viewport_size: Vector2
var processing: bool = false
var frame_count: int = 0
var last_time: float = 0.0
var current_fps: float = 0.0
const TARGET_FPS: int = 15
var detection_boxes = []
var detection_labels = []
var coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

func _ready() -> void:
    if not web_camera_manager.start_camera():
        print("Failed to start camera")
        return
    
    viewport_size = get_viewport_rect().size
    get_viewport().size_changed.connect(_on_viewport_size_changed)
    
    print("Initialization complete")
    
    if not segmentation.load_model("res://models/yolact_edge_mobilenetv2_550x550.onnx"):
        print("Failed to load segmentation model")

func _process(_delta: float) -> void:
    if not web_camera_manager.capture_frame():
        print("Failed to capture frame")
        return
    
    _update_display_transform()
        
    var texture = web_camera_manager.get_texture()
    
    if texture:
        camera_display.texture = texture
        
        frame_count += 1
        var current_time = Time.get_ticks_msec() / 1000.0
        if current_time - last_time >= 1.0:
            current_fps = frame_count / (current_time - last_time)
            frame_count = 0
            last_time = current_time
            
        var frames_per_second = max(1, int(Engine.get_frames_per_second() / TARGET_FPS))
        if Engine.get_frames_drawn() % frames_per_second == 0:
            if not processing and segmentation and segmentation.is_ready():
                processing = true
                process_current_frame()
                processing = false
    
    queue_redraw()

func _draw():
    if not camera_display.texture:
        return
        
    for i in range(len(detection_boxes)):
        var rect = detection_boxes[i]
        var label = detection_labels[i]
        
        var class_id = label["class_id"]
        var color = segmentation.get_class_color(class_id)
        
        var outline_color = Color(
            max(0, color.r - 0.3),
            max(0, color.g - 0.3),
            max(0, color.b - 0.3),
            1.0
        )
        
        var texture_size = camera_display.texture.get_size()
        
        var scaled_pos = Vector2(
            rect.position.x * camera_display.scale.x,
            rect.position.y * camera_display.scale.y
        )
        var scaled_size = Vector2(
            rect.size.x * camera_display.scale.x,
            rect.size.y * camera_display.scale.y
        )
        
        var transformed_rect = Rect2(
            camera_display.position + scaled_pos - (texture_size * camera_display.scale / 2),
            scaled_size
        )
        
        draw_rect(transformed_rect.grow(2), Color.BLACK, false, 5.0)
        draw_rect(transformed_rect, color, false, 3.0)
        
        var font_size = 16
        var label_text = label["class_name"] + " " + str(int(label["score"] * 100)) + "%"
        var label_size = Vector2(label_text.length() * font_size * 0.6, font_size * 1.5)
        var label_pos = transformed_rect.position - Vector2(0, label_size.y)
        
        if label_pos.x < 0:
            label_pos.x = 0
        if label_pos.y < 0:
            label_pos.y = 0
            
        draw_rect(Rect2(label_pos, label_size).grow(2), Color.BLACK, true)
        draw_rect(Rect2(label_pos, label_size), color, true)
        
        draw_string(ThemeDB.fallback_font, label_pos + Vector2(5, font_size), label_text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color.WHITE)

func _on_viewport_size_changed() -> void:
    viewport_size = get_viewport_rect().size
    _update_display_transform()

func _update_display_transform() -> void:
    if not camera_display.texture:
        return
    
    var texture_size = camera_display.texture.get_size()
    var scale_factor = min(viewport_size.x / texture_size.x, viewport_size.y / texture_size.y)
    
    camera_display.scale = Vector2(scale_factor, scale_factor)
    camera_display.position = viewport_size / 2
    
    if mask_display.texture:
        mask_display.scale = camera_display.scale
        mask_display.position = camera_display.position

func process_current_frame() -> void:
    var image = web_camera_manager.get_image()
    if not image:
        print("Failed to get image from camera")
        return
    
    print("Camera image: " + str(image.get_size()) + " format: " + str(image.get_format()))
    
    segmentation.set_current_frame(image)
    print("Frame set, processing objects...")
    
    var result = segmentation.process_all_objects()
    print("process_all_objects returned: " + str(result))
    
    detection_boxes.clear()
    detection_labels.clear()
    
    for child in mask_display.get_children():
        child.queue_free()
    
    if result:
        var mask_image = segmentation.get_combined_mask()
        
        if mask_image:
            print("Processing successful. Mask dimensions: " + str(mask_image.get_size()) + " format: " + str(mask_image.get_format()))
            
            var mask_texture = ImageTexture.create_from_image(mask_image)
            if mask_texture:
                mask_display.texture = mask_texture
                
                if mask_display.z_index <= camera_display.z_index:
                    mask_display.z_index = camera_display.z_index + 1
            
            var detection_count = segmentation.get_detection_count()
            print("Detected " + str(detection_count) + " objects")
            
            var boxes = segmentation.get_detection_boxes()
            var scores = segmentation.get_detection_scores()
            var classes = segmentation.get_detection_classes()
            
            for i in range(detection_count):
                if i < boxes.size():
                    var detection = segmentation.get_detection_at_index(i)
                    
                    detection_boxes.append(boxes[i])
                    
                    var label_info = {
                        "class_id": classes[i],
                        "class_name": get_class_name(classes[i]),
                        "score": scores[i]
                    }
                    detection_labels.append(label_info)
        else:
            print("No valid mask was returned")
    else:
        print("Object processing failed - attempting to diagnose...")
        
        var is_ready = segmentation.is_ready()
        print("Segmentation ready state: " + str(is_ready))
        
func get_class_name(class_id: int) -> String:
    if class_id >= 0 and class_id < coco_classes.size():
        return coco_classes[class_id]
    else:
        return "Unknown (Class " + str(class_id) + ")"

func _on_threshold_changed(value: float) -> void:
    if segmentation and segmentation.is_ready():
        segmentation.set_detection_threshold(value)
        print("Detection threshold set to: " + str(value))
