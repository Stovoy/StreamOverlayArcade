extends Node2D

@export var droplet_count_range: Vector2i = Vector2i(5, 20)
@export var spawn_interval_range: Vector2 = Vector2(0.01, 0.02)
@export var droplet_size_range: Vector2 = Vector2(5.0, 50.0)
@export var droplet_velocity_range: Vector2 = Vector2(20.0, 100.0)
@export var droplet_bounce: float = 0.5
@export var droplet_friction: float = 0.1
@export var droplet_gravity_scale: float = 1.0
@export var droplet_lifetime: float = 10.0

var droplet_scene: PackedScene
var spawn_timer: Timer
var viewport_size: Vector2
var rng = RandomNumberGenerator.new()

func _ready():
    rng.randomize()
    
    # Create the droplet scene at runtime
    droplet_scene = _create_droplet_scene()
    
    # Setup spawn timer
    spawn_timer = Timer.new()
    spawn_timer.one_shot = true
    spawn_timer.timeout.connect(_on_spawn_timer_timeout)
    add_child(spawn_timer)
    
    # Start spawning
    _reset_spawn_timer()
    
    # Get viewport size for placement
    viewport_size = get_viewport_rect().size
    get_viewport().size_changed.connect(_on_viewport_size_changed)

func _create_droplet_scene() -> PackedScene:
    # Create a template for the droplet
    var scene_root = RigidBody2D.new()
    scene_root.collision_layer = 1
    scene_root.collision_mask = 1
    scene_root.mass = 1.0
    
    # Create a physics material for bounce and friction
    var physics_material = PhysicsMaterial.new()
    physics_material.bounce = droplet_bounce
    physics_material.friction = droplet_friction
    scene_root.physics_material_override = physics_material
    
    scene_root.gravity_scale = droplet_gravity_scale
    
    # Add a collision shape
    var collision = CollisionShape2D.new()
    var shape = CircleShape2D.new()
    shape.radius = 10.0  # Default radius, will be scaled
    collision.shape = shape
    scene_root.add_child(collision)
    
    # Create a packed scene
    var packed_scene = PackedScene.new()
    packed_scene.pack(scene_root)
    return packed_scene

func _on_spawn_timer_timeout():
    spawn_droplets()
    _reset_spawn_timer()

func _reset_spawn_timer():
    var next_spawn_time = rng.randf_range(spawn_interval_range.x, spawn_interval_range.y)
    spawn_timer.start(next_spawn_time)

func spawn_droplets():
    var count = rng.randi_range(droplet_count_range.x, droplet_count_range.y)
    
    for i in range(count):
        var droplet = droplet_scene.instantiate() as RigidBody2D
        
        # Set random size
        var size = rng.randf_range(droplet_size_range.x, droplet_size_range.y)
        
        # Find collision shape - check if children exist first
        if droplet.get_child_count() > 0:
            var collision_shape = droplet.get_child(0) as CollisionShape2D
            if collision_shape and collision_shape.shape:
                var circle_shape = collision_shape.shape as CircleShape2D
                circle_shape.radius = size / 2.0
        else:
            # If no collision shape exists, create one
            var collision = CollisionShape2D.new()
            var shape = CircleShape2D.new()
            shape.radius = size / 2.0
            collision.shape = shape
            droplet.add_child(collision)
        
        # Set random properties
        droplet.mass = size / 20.0
        
        # Create custom physics material with randomized properties
        var physics_material = PhysicsMaterial.new()
        physics_material.bounce = droplet_bounce + rng.randf_range(-0.1, 0.1)
        physics_material.friction = droplet_friction + rng.randf_range(-0.05, 0.05)
        droplet.physics_material_override = physics_material
        
        # Set initial position at top of screen
        var x_pos = rng.randf_range(0, viewport_size.x)
        droplet.position = Vector2(x_pos, -size)
        
        # Set initial velocity
        var velocity = Vector2(
            rng.randf_range(-droplet_velocity_range.x, droplet_velocity_range.x),
            rng.randf_range(droplet_velocity_range.x, droplet_velocity_range.y)
        )
        droplet.linear_velocity = velocity
        
        # Create visual representation with random color tint
        var polygon = create_droplet_polygon(size)
        droplet.add_child(polygon)
        
        # Add to scene first (important for timer to work properly)
        add_child(droplet)
        
        # Set up auto-destroy after lifetime
        var timer = Timer.new()
        timer.one_shot = true
        timer.wait_time = droplet_lifetime + rng.randf_range(-2.0, 2.0)
        droplet.add_child(timer) # Add timer to the droplet
        timer.timeout.connect(func(): droplet.queue_free())
        timer.start()

func create_droplet_polygon(size: float) -> Polygon2D:
    var polygon = Polygon2D.new()
    
    # Create water-like shape with randomized vertices
    var vertices = []
    var vertex_count = rng.randi_range(8, 12)
    var base_radius = size / 2.0
    
    for i in range(vertex_count):
        var angle = 2.0 * PI * i / vertex_count
        # Add some randomness to the radius to create irregular shape
        var radius = base_radius * rng.randf_range(0.8, 1.2)
        vertices.append(Vector2(cos(angle) * radius, sin(angle) * radius))
    
    polygon.polygon = vertices
    
    # Create a water-like appearance
    var blue_intensity = rng.randf_range(0.5, 0.9)
    var alpha = rng.randf_range(0.5, 0.8)
    polygon.color = Color(0.0, blue_intensity, 1.0, alpha)
    
    return polygon

func _on_viewport_size_changed():
    viewport_size = get_viewport_rect().size
