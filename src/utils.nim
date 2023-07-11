import vmath
proc cStringToString*(arr: openArray[char]): string =
    for c in items(arr):
        if c != '\0':
            result = result & c

type
    Vertex* = object
        pos*: Vec2
        color*: Vec3

proc vertex*(pos: Vec2, color: Vec3): Vertex =
    result = Vertex(pos : pos, color: color)