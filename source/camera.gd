extends Node2D

@onready var web_camera_manager = $WebCameraManager
@onready var camera_display = $CameraDisplay
var viewport_size: Vector2

func _ready() -> void:
    web_camera_manager.start_camera()
    viewport_size = get_viewport_rect().size
    get_viewport().size_changed.connect(_on_viewport_size_changed)

func _process(delta: float) -> void:
    web_camera_manager.capture_frame()
    var texture = web_camera_manager.get_texture()
    
    if texture:
        camera_display.texture = texture
        _update_display_transform()

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
