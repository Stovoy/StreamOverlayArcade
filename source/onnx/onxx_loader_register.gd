@tool
extends Node

func _ready():
    ResourceLoader.add_resource_format_loader(OnnxLoader.new())
