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

var detection_labels = []
var physics_bodies = []

var inference_thread: Thread = null
var is_inference_running: bool = false
var latest_inference_result: Dictionary = {}
var latest_camera_frame: Image = null
var min_capture_interval_ms: int = 100
var last_capture_time: int = 0
var initialized: bool = false

func _ready() -> void:
    if not web_camera_manager.start_camera():
        print("Failed to start camera")
        return
    
    viewport_size = get_viewport_rect().size
    get_viewport().size_changed.connect(_on_viewport_size_changed)
    
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
    call_deferred("_on_inference_complete", result)

func _on_inference_complete(result: Dictionary) -> void:
    latest_inference_result = result
    is_inference_running = false
    if inference_thread:
        inference_thread.wait_to_finish()
        inference_thread = null

func process_image_in_thread(image: Image) -> Dictionary:
    if not segmentation or not segmentation.is_ready():
        return {"success": false}
    
    # Pass the frame directly to the segmentation module
    segmentation.set_current_frame(image)
    
    # Get all results in a single call
    var result_dict = segmentation.process_segmentation()
    
    if result_dict.size() > 0:
        # The mask image is directly in the result dictionary
        var mask_image = result_dict.get("mask")
        
        # Get detections array from the result
        var detections = result_dict.get("detections", [])
        var detection_count = detections.size()
        
        # Extract scores and classes from detections
        var scores = PackedFloat32Array()
        var classes = PackedInt32Array()
        
        for i in range(detection_count):
            var detection = detections[i]
            scores.append(detection.get("score", 1.0))
            classes.append(detection.get("class_id", 0))
            
        return {
            "success": true,
            "mask_image": mask_image.duplicate() if mask_image else null,
            "detection_count": detection_count,
            "scores": scores,
            "classes": classes,
            "detections": detections
        }
    
    return {"success": false}

func update_detection_results(results: Dictionary) -> void:
    if not results.has("success") or not results["success"]:
        return
    
    detection_labels.clear()
    
    # Clear existing physics bodies
    for body in physics_bodies:
        if is_instance_valid(body):
            body.queue_free()
    physics_bodies.clear()
    
    for child in mask_display.get_children():
        child.queue_free()
    
    var mask_image = results["mask_image"]
    if not mask_image:
        print("No valid mask was returned")
        return

    var detection_count = results["detection_count"]
    var scores = results["scores"]
    var classes = results["classes"]
    var segmentation_masks = results.get("segmentation_masks", [])
    var detections = results.get("detections", [])
    
    var mask_texture = ImageTexture.create_from_image(mask_image)
    mask_display.texture = mask_texture
    
    if mask_display.z_index <= camera_display.z_index:
        mask_display.z_index = camera_display.z_index + 1
    
    for i in range(detection_count):
        var detection = detections[i]
        var class_id = detection.get("class_id", 0)
        var _class_name = detection.get("class_name", str(class_id))
        var score = detection.get("score", 1.0)
        var color = detection.get("color", Color(1, 0, 0))
        
        var label_info = {
            "class_id": class_id,
            "class_name": _class_name,
            "score": score,
            "color": color
        }
        detection_labels.append(label_info)
        
        if class_id == 15:
            create_physics_body_from_mask(mask_image, class_id, color)

func create_physics_body_from_mask(mask_image: Image, class_id: int, color: Color = Color(1, 0, 0)) -> void:
    var static_body = StaticBody2D.new()
    static_body.position = camera_display.position
    var collision_polygon = create_collision_polygon_from_mask(mask_image, color)
    
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

func create_collision_polygon_from_mask(mask_image: Image, color: Color = Color(1, 0, 0), downsample_factor: float = 4.0) -> CollisionPolygon2D:
    var new_width = max(1, int(mask_image.get_width() / downsample_factor))
    var new_height = max(1, int(mask_image.get_height() / downsample_factor))
    mask_image.resize(new_width, new_height, Image.INTERPOLATE_NEAREST)
    
    for y in range(new_height):
        for x in range(new_width):
            var pixel = mask_image.get_pixel(x, y)
            if pixel.a > 0:
                pixel.a = 1
            if pixel == color:
                mask_image.set_pixel(x, y, Color(1, 1, 1))
            else:
                mask_image.set_pixel(x, y, Color(0, 0, 0))
    
    var all_polygons = marching_squares(mask_image)
    if all_polygons.size() == 0:
        return CollisionPolygon2D.new()
    
    var biggest_polygon = all_polygons[0]
    var biggest_area = get_polygon_area(biggest_polygon)
    for polygon in all_polygons:
        var area = get_polygon_area(polygon)
        if area > biggest_area:
            biggest_polygon = polygon
            biggest_area = area
    
    var simplified = simplify_polygon(biggest_polygon, 2.0)
    for i in range(simplified.size()):
        simplified[i] *= downsample_factor
    
    var collision_polygon = CollisionPolygon2D.new()
    collision_polygon.polygon = simplified
    return collision_polygon

func marching_squares(src_image: Image) -> Array:
    var width = src_image.get_width()
    var height = src_image.get_height()
    var data = []
    
    for y in range(height):
        var row = []
        for x in range(width):
            var value = src_image.get_pixel(x, y).r
            row.append(1 if value > 0 else 0)
        data.append(row)
    
    var edges = []
    for y in range(height - 1):
        for x in range(width - 1):
            var v0 = data[y][x]
            var v1 = data[y][x + 1]
            var v2 = data[y + 1][x + 1]
            var v3 = data[y + 1][x]
            var cell_index = v0*1 + v1*2 + v2*4 + v3*8
            var cell_edges = get_marching_square_edges(x, y, v0, v1, v2, v3)
            for edge in cell_edges:
                edges.append(edge)
    
    var polygons = assemble_polygons(edges)
    return polygons

func get_marching_square_edges(x: int, y: int, v0: int, v1: int, v2: int, v3: int) -> Array:
    var cell_index = v0*1 + v1*2 + v2*4 + v3*8
    match cell_index:
        0, 15: return []
        1: return [[Vector2(x, y+0.5), Vector2(x+0.5, y)]]
        2: return [[Vector2(x+0.5, y), Vector2(x+1, y+0.5)]]
        3: return [[Vector2(x, y+0.5), Vector2(x+1, y+0.5)]]
        4: return [[Vector2(x+1, y+0.5), Vector2(x+0.5, y+1)]]
        5: return [
            [Vector2(x, y+0.5), Vector2(x+0.5, y)],
            [Vector2(x+1, y+0.5), Vector2(x+0.5, y+1)]
        ]
        6: return [[Vector2(x+0.5, y), Vector2(x+0.5, y+1)]]
        7: return [[Vector2(x, y+0.5), Vector2(x+0.5, y+1)]]
        8: return [[Vector2(x+0.5, y+1), Vector2(x, y+0.5)]]
        9: return [[Vector2(x+0.5, y), Vector2(x+0.5, y+1)]]
        10: return [
            [Vector2(x+0.5, y), Vector2(x+1, y+0.5)],
            [Vector2(x+0.5, y+1), Vector2(x, y+0.5)]
        ]
        11: return [[Vector2(x+0.5, y+1), Vector2(x+1, y+0.5)]]
        12: return [[Vector2(x, y+0.5), Vector2(x+1, y+0.5)]]
        13: return [[Vector2(x+0.5, y), Vector2(x+1, y+0.5)]]
        14: return [[Vector2(x, y+0.5), Vector2(x+0.5, y)]]
        _: return []

func assemble_polygons(edge_list: Array) -> Array:
    var adjacency = {}
    for edge in edge_list:
        var a = edge[0]
        var b = edge[1]
        var a_key = str(int(a.x*1000)) + "_" + str(int(a.y*1000))
        var b_key = str(int(b.x*1000)) + "_" + str(int(b.y*1000))
        
        if not adjacency.has(a_key):
            adjacency[a_key] = []
        if not adjacency.has(b_key):
            adjacency[b_key] = []
            
        adjacency[a_key].append(b)
        adjacency[b_key].append(a)
    
    var visited_edges = {}
    var polygons = []
    
    for key in adjacency.keys():
        if adjacency[key].size() == 0:
            continue
            
        for neighbor in adjacency[key]:
            var edge_id = key + "_" + str(int(neighbor.x*1000)) + "_" + str(int(neighbor.y*1000))
            if visited_edges.has(edge_id):
                continue
                
            var chain = []
            var start_key = key
            var current_key = key
            var next_point = neighbor
            
            while true:
                var coords = current_key.split("_")
                var cx = float(coords[0]) / 1000.0
                var cy = float(coords[1]) / 1000.0
                chain.append(Vector2(cx, cy))
                
                var edge_key = current_key + "_" + str(int(next_point.x*1000)) + "_" + str(int(next_point.y*1000))
                visited_edges[edge_key] = true
                
                current_key = str(int(next_point.x*1000)) + "_" + str(int(next_point.y*1000))
                if current_key == start_key:
                    break
                    
                var found_next = false
                for n in adjacency[current_key]:
                    var n_key = str(int(n.x*1000)) + "_" + str(int(n.y*1000))
                    var next_edge_key = current_key + "_" + n_key
                    
                    if not visited_edges.has(next_edge_key) or n_key == start_key:
                        next_point = n
                        found_next = true
                        break
                        
                if not found_next:
                    chain.append(next_point)
                    break
            
            if chain.size() > 2:
                polygons.append(chain)
    
    return polygons

func get_polygon_area(polygon: Array) -> float:
    var area = 0.0
    var count = polygon.size()
    for i in range(count):
        var j = (i + 1) % count
        area += polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y
    return abs(area) * 0.5

func simplify_polygon(points: Array, tolerance: float, depth: int = 0) -> Array:
    if depth > 8:
        return points
    if points.size() < 3:
        return points
        
    var dmax = 0.0
    var index = 0
    var end = points.size() - 1
    
    for i in range(1, end):
        var dist_val = perpendicular_distance(points[i], points[0], points[end])
        if dist_val > dmax:
            index = i
            dmax = dist_val
            
    if dmax > tolerance:
        var left_segment = simplify_polygon(points.slice(0, index + 1), tolerance, depth + 1)
        var right_segment = simplify_polygon(points.slice(index, points.size()), tolerance, depth + 1)
        
        var result = []
        for p in left_segment:
            result.append(p)
        if result.size() > 0:
            result.pop_back()
        for p in right_segment:
            result.append(p)
            
        return result
    else:
        return [points[0], points[end]]

func perpendicular_distance(point: Vector2, line_start: Vector2, line_end: Vector2) -> float:
    if line_start == line_end:
        return point.distance_to(line_start)
    var numerator = abs((line_end.y - line_start.y) * point.x - (line_end.x - line_start.x) * point.y + line_end.x * line_start.y - line_end.y * line_start.x)
    var denominator = line_start.distance_to(line_end)
    return numerator / (denominator if denominator != 0 else 0.00001)

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
