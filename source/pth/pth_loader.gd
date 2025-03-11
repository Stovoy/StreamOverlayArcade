@tool
extends ResourceFormatLoader
class_name PthLoader

func _exists(path: String) -> bool:
    var f = FileAccess.open(path, FileAccess.READ)
    if f:
        f.close()
        return true
    return false

func _get_recognized_extensions() -> PackedStringArray:
    return PackedStringArray(["pth"])

func _get_resource_type(path: String) -> String:
    if path.to_lower().ends_with(".pth"):
        return "PthResource"
    return ""

func _handles_type(type: StringName) -> bool:
    return type == "PthResource"

func _load(path: String, _original_path: String, _use_sub_threads: bool, _cache_mode: int) -> Variant:
    var f = FileAccess.open(path, FileAccess.READ)
    if f == null:
        return ERR_CANT_OPEN
    var res = PthResource.new()
    res.data = f.get_buffer(f.get_length())
    f.close()
    return res
