import vmath
import std/hashes
proc cStringToString*(arr: openArray[char]): string =
    for c in items(arr):
        if c != '\0':
            result = result & c

type
    Vertex* = object
        pos*: Vec3
        color*: Vec3
        texCoord*: Vec2

proc vertex*(pos: Vec3, color: Vec3, texCoord: Vec2): Vertex =
    result = Vertex(pos : pos, color: color, texCoord: texCoord)

proc `==`*(a, b: Vertex): bool =
    a.pos == b.pos and a.color == b.color and a.texCoord == b.texCoord

proc hash*(x: Vertex): Hash =
    result = x.pos.hash !& x.color.hash !& x.texCoord.hash # '!&' used to mix hashes

type
    UniformBufferObject* = object
        model*: Mat4
        view*: Mat4
        proj*: Mat4

const vulMat4Y = mat4( # Conversion matrix from OpenGL's Y[0..1] to Vulkan's Y[0..-1]
    1.0, 0.0,  0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0,  1.0, 0.0,
    0.0, 0.0,  0.0, 1.0,
)
const vulMat4Z = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
)

proc toVulY*[T](proj :GMat4[T]) :GMat4[T]=  proj*vulMat4Y
proc toVulZ*[T](proj :GMat4[T]) :GMat4[T]=  proj*vulMat4Z
