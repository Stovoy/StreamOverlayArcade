extends Node2D

@onready var web_camera_manager = $WebCameraManager
@onready var camera_display = $CameraDisplay
@onready var mask_display = $MaskDisplay
@onready var segmentation = $MediaPipeSegmentation
@onready var physics_bodies_container = $PhysicsBodiesContainer

var viewport_size: Vector2
var frame_count: int = 0
var last_time: float = 0.0
var current_fps: float = 0.0

var detection_boxes = []
var detection_labels = []
var physics_bodies = []

var inference_thread: Thread = null
var is_inference_running: bool = false
var latest_inference_result: Dictionary = {}
var latest_camera_frame: Image = null
var min_capture_interval_ms: int = 100
var last_capture_time: int = 0
var initialized: bool = false

var coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

func _ready() -> void:
    if not web_camera_manager.start_camera():
        print("Failed to start camera")
        return
    
    viewport_size = get_viewport_rect().size
    get_viewport().size_changed.connect(_on_viewport_size_changed)
    
    var model_path = "res://models/selfie_segmenter.tflite"
    print("Loading model from: " + model_path)
    
    if not segmentation.load_model(model_path):
        print("Failed to load segmentation model: " + model_path)
    else:
        print("Model loaded successfully!")
    
    initialized = true

func _process(_delta: float) -> void:
    if not initialized:
        return
    if not web_camera_manager.capture_frame():
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
        
        var current_time_ms = Time.get_ticks_msec()
        if current_time_ms - last_capture_time >= min_capture_interval_ms:
            var image = web_camera_manager.get_image()
            if image:
                latest_camera_frame = image.duplicate()
                last_capture_time = current_time_ms
    
    if latest_inference_result:
        update_detection_results(latest_inference_result)
        latest_inference_result = {}
    
    if not is_inference_running and latest_camera_frame:
        is_inference_running = true
        inference_thread = Thread.new()
        inference_thread.start(Callable(self, "_inference_thread_function").bind(latest_camera_frame.duplicate()))
        latest_camera_frame = null

func _inference_thread_function(frame: Image) -> void:
    var result = process_image_in_thread(frame)
    # Use call_deferred to safely return to main thread without blocking
    call_deferred("_on_inference_complete", result)

func _on_inference_complete(result: Dictionary) -> void:
    latest_inference_result = result
    is_inference_running = false
    inference_thread.wait_to_finish()
    inference_thread = null

func process_image_in_thread(image: Image) -> Dictionary:
    if not segmentation or not segmentation.is_ready():
        return {"success": false}
    
    segmentation.set_current_frame(image)
    var result = segmentation.process_all_objects()
    
    if result:
        var mask_image = segmentation.get_combined_mask()
        var detection_count = segmentation.get_detection_count()
        var boxes = segmentation.get_detection_boxes()
        var scores = segmentation.get_detection_scores()
        var classes = segmentation.get_detection_classes()
        
        # Get individual masks for each detection
        var segmentation_masks = []
        for i in range(detection_count):
            var detection_dict = segmentation.get_detection_at_index(i)
            if detection_dict.has("mask"):
                segmentation_masks.append(detection_dict["mask"])
            else:
                segmentation_masks.append(null)
        
        return {
            "success": true,
            "mask_image": mask_image.duplicate(),
            "detection_count": detection_count,
            "boxes": boxes.duplicate(),
            "scores": scores.duplicate(),
            "classes": classes.duplicate(),
            "segmentation_masks": segmentation_masks
        }
    
    return {"success": false}

func update_detection_results(results: Dictionary) -> void:
    if not results.has("success") or not results["success"]:
        return
    
    detection_boxes.clear()
    detection_labels.clear()
    
    # Clear existing physics bodies
    for body in physics_bodies:
        if is_instance_valid(body):
            body.queue_free()
    physics_bodies.clear()
    
    for child in mask_display.get_children():
        child.queue_free()
    
    var mask_image = results["mask_image"]
    if mask_image:
        var mask_texture = ImageTexture.create_from_image(mask_image)
        if mask_texture:
            mask_display.texture = mask_texture
            
            if mask_display.z_index <= camera_display.z_index:
                mask_display.z_index = camera_display.z_index + 1
        
        var detection_count = results["detection_count"]
        
        var boxes = results["boxes"]
        var scores = results["scores"]
        var classes = results["classes"]
        var segmentation_masks = results.get("segmentation_masks", [])
        
        for i in range(detection_count):
            if i < boxes.size():
                detection_boxes.append(boxes[i])
                
                var label_info = {
                    "class_id": classes[i],
                    "class_name": get_class_name(classes[i]),
                    "score": scores[i]
                }
                detection_labels.append(label_info)
                
                # Create physics body from mask only for person class (class_id 0)
                if i < segmentation_masks.size() and segmentation_masks[i] != null and classes[i] == 0:
                    create_physics_body_from_mask(segmentation_masks[i], boxes[i], classes[i])
    else:
        print("No valid mask was returned")

func create_physics_body_from_mask(mask_image: Image, box: Rect2, class_id: int) -> void:
    var static_body = StaticBody2D.new()
    static_body.position = camera_display.position
    var color = segmentation.get_class_color(class_id)
    var collision_polygon = create_collision_polygon_from_mask(mask_image, box)
    
    if collision_polygon:
        for i in range(collision_polygon.polygon.size()):
            collision_polygon.polygon[i] *= camera_display.scale
        
        var texture_size = camera_display.texture.get_size()
        collision_polygon.position = -texture_size * camera_display.scale / 2
        
        # Create an Area2D to detect and push out overlapping objects
        var area = Area2D.new()
        area.name = "OverlapArea"
        var area_collision = CollisionPolygon2D.new()
        area_collision.polygon = collision_polygon.polygon.duplicate()
        area_collision.position = collision_polygon.position
        area.add_child(area_collision)
        
        # Connect body entered signal
        area.body_entered.connect(_on_body_entered.bind(area))
        
        static_body.add_child(collision_polygon)
        static_body.add_child(area)
        physics_bodies_container.add_child(static_body)
        physics_bodies.append(static_body)
        
        # Immediately check for and push out any existing overlapping bodies
        call_deferred("_check_overlapping_bodies", area)

func _check_overlapping_bodies(area: Area2D) -> void:
    # Wait a physics frame to ensure bodies are properly detected
    await get_tree().physics_frame
    
    var overlapping_bodies = area.get_overlapping_bodies()
    for body in overlapping_bodies:
        _push_body_out(body, area)

func _on_body_entered(body: Node2D, area: Area2D) -> void:
    _push_body_out(body, area)

func _push_body_out(body: Node2D, area: Area2D) -> void:
    # Skip pushing out the static body itself or its parent
    if body == area.get_parent() or body == physics_bodies_container:
        return
        
    # Only push out RigidBody2D or CharacterBody2D objects
    if not (body is RigidBody2D or body is CharacterBody2D):
        return
        
    var body_center = body.global_position
    
    # Check if the body is actually inside the polygon
    var polygon_points = []
    var collision_poly = null
    
    for child in area.get_children():
        if child is CollisionPolygon2D:
            collision_poly = child as CollisionPolygon2D
            for i in range(collision_poly.polygon.size()):
                var global_point = area.global_position + collision_poly.position + collision_poly.polygon[i]
                polygon_points.append(global_point)
            break
    
    if polygon_points.size() == 0:
        return
    
    # Check if body center is inside the polygon
    if not _is_point_in_polygon(body_center, polygon_points):
        return  # Body is not inside, don't push
    
    # Find the closest edge point to teleport to
    var closest_point = _find_closest_edge_point(body_center, polygon_points)
    
    # Calculate the minimum distance needed to move outside
    var edge_direction = (closest_point - body_center).normalized()
    var body_radius = 20.0  # Approximate radius for most objects
    
    # Calculate new position just outside the shape
    var new_position = closest_point + (edge_direction * body_radius)
    
    # Apply the new position directly
    if body is RigidBody2D:
        # For RigidBody2D we need to reset forces too
        body.global_position = new_position
        body.linear_velocity = Vector2.ZERO
    elif body is CharacterBody2D:
        body.global_position = new_position
        body.velocity = Vector2.ZERO

func _is_point_in_polygon(point: Vector2, polygon_points: Array) -> bool:
    # Ray casting algorithm to determine if a point is inside a polygon
    var inside = false
    var j = polygon_points.size() - 1
    
    for i in range(polygon_points.size()):
        if ((polygon_points[i].y > point.y) != (polygon_points[j].y > point.y)) and \
           (point.x < polygon_points[i].x + (polygon_points[j].x - polygon_points[i].x) * (point.y - polygon_points[i].y) / (polygon_points[j].y - polygon_points[i].y)):
            inside = not inside
        j = i
    
    return inside

func _find_closest_edge_point(point: Vector2, polygon_points: Array) -> Vector2:
    var closest_point = polygon_points[0]
    var closest_distance = point.distance_squared_to(closest_point)
    
    for i in range(polygon_points.size()):
        var start = polygon_points[i]
        var end = polygon_points[(i + 1) % polygon_points.size()]
        
        var closest_on_segment = _closest_point_on_segment(point, start, end)
        var distance = point.distance_squared_to(closest_on_segment)
        
        if distance < closest_distance:
            closest_distance = distance
            closest_point = closest_on_segment
            
    return closest_point

func _closest_point_on_segment(point: Vector2, line_start: Vector2, line_end: Vector2) -> Vector2:
    var line_vector = line_end - line_start
    var length_squared = line_vector.length_squared()
    
    if length_squared == 0:
        return line_start
        
    var t = max(0, min(1, (point - line_start).dot(line_vector) / length_squared))
    return line_start + t * line_vector

func create_collision_polygon_from_mask(mask_image: Image, box: Rect2) -> CollisionPolygon2D:
    var downsample_factor = 4.0
    var original_width = mask_image.get_width()
    var original_height = mask_image.get_height()
    var new_width = max(1, int(original_width / downsample_factor))
    var new_height = max(1, int(original_height / downsample_factor))

    mask_image.resize(new_width, new_height, Image.INTERPOLATE_NEAREST)

    var width = mask_image.get_width()
    var height = mask_image.get_height()
    var binary = []

    for y in range(height):
        for x in range(width):
            binary.append(1 if mask_image.get_pixel(x, y).a > 0 else 0)
            
    var start_point = null
    for y in range(height):
        for x in range(width):
            var index = y * width + x
            if binary[index] == 1:
                var is_edge = false
                for offset in [Vector2(-1, 0), Vector2(1, 0), Vector2(0, -1), Vector2(0, 1)]:
                    var nx = x + int(offset.x)
                    var ny = y + int(offset.y)
                    if nx < 0 or nx >= width or ny < 0 or ny >= height or binary[ny * width + nx] == 0:
                        is_edge = true
                        break
                if is_edge:
                    start_point = Vector2(x, y)
                    break
        if start_point != null:
            break

    if start_point == null:
        return CollisionPolygon2D.new()

    var contour = [start_point]
    var current = start_point
    var current_direction = Vector2(1, 0)
    var directions = [Vector2(1, 0), Vector2(1, 1), Vector2(0, 1), Vector2(-1, 1), Vector2(-1, 0), Vector2(-1, -1), Vector2(0, -1), Vector2(1, -1)]
    var max_iterations = 10000
    var iterations = 0

    while iterations < max_iterations:
        var found_next = false
        var start_index = directions.find(current_direction)
        if start_index == -1:
            start_index = 0
        var search_index = (start_index + 7) % 8
        for i in range(8):
            var direction_vector = directions[(search_index + i) % 8]
            var next_point = current + direction_vector
            var nx = int(next_point.x)
            var ny = int(next_point.y)
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            var neighbor_index = ny * width + nx
            if binary[neighbor_index] == 1:
                var is_edge = false
                for offset in [Vector2(-1, 0), Vector2(1, 0), Vector2(0, -1), Vector2(0, 1)]:
                    var check_x = nx + int(offset.x)
                    var check_y = ny + int(offset.y)
                    if check_x < 0 or check_x >= width or check_y < 0 or check_y >= height or binary[check_y * width + check_x] == 0:
                        is_edge = true
                        break
                if is_edge:
                    current = next_point
                    current_direction = direction_vector
                    if current == start_point:
                        iterations = max_iterations
                        found_next = false
                        break
                    contour.append(current)
                    found_next = true
                    break

        if not found_next:
            break

        iterations += 1

    var simplified_contour = simplify_polygon(contour, 1.0)
    for i in range(simplified_contour.size()):
        simplified_contour[i] *= downsample_factor

    var collision_polygon = CollisionPolygon2D.new()
    collision_polygon.polygon = simplified_contour
    return collision_polygon

func simplify_polygon(points: Array, tolerance: float) -> Array:
    if points.size() < 3:
        return points
    var dmax = 0.0
    var index = 0
    for i in range(1, points.size() - 1):
        var distance_value = perpendicular_distance(points[i], points[0], points[points.size() - 1])
        if distance_value > dmax:
            index = i
            dmax = distance_value
    if dmax > tolerance:
        var left_segment = simplify_polygon(points.slice(0, index + 1), tolerance)
        var right_segment = simplify_polygon(points.slice(index, points.size()), tolerance)
        var result = left_segment.duplicate()
        result.remove_at(result.size() - 1)
        for point in right_segment:
            result.append(point)
        return result
    else:
        return [points[0], points[points.size() - 1]]

func perpendicular_distance(point: Vector2, line_start: Vector2, line_end: Vector2) -> float:
    if line_start == line_end:
        return point.distance_to(line_start)
    var numerator = abs((line_end.y - line_start.y) * point.x - (line_end.x - line_start.x) * point.y + line_end.x * line_start.y - line_end.y * line_start.x)
    var denominator = line_start.distance_to(line_end)
    return numerator / denominator

func _draw() -> void:
    if not camera_display.texture:
        return
        
    for i in range(len(detection_boxes)):
        var rect = detection_boxes[i]
        var label = detection_labels[i]
        
        var class_id = label["class_id"]
        var color = segmentation.get_class_color(class_id)
        
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
    
    # Update positions of all physics bodies
    for body in physics_bodies:
        if is_instance_valid(body):
            body.position = camera_display.position

func get_class_name(class_id: int) -> String:
    if class_id >= 0 and class_id < coco_classes.size():
        return coco_classes[class_id]
    return "Unknown (Class " + str(class_id) + ")"

func _on_threshold_changed(value: float) -> void:
    if segmentation and segmentation.is_ready():
        segmentation.set_detection_threshold(value)
        print("Detection threshold set to: " + str(value))
