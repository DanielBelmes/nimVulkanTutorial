import vmath
proc cStringToString*(arr: openArray[char]): string =
    for c in items(arr):
        if c != '\0':
            result = result & c

type
    Vertex* = object
        pos*: Vec2
        color*: Vec3
        texCoord*: Vec2

proc vertex*(pos: Vec2, color: Vec3, texCoord: Vec2): Vertex =
    result = Vertex(pos : pos, color: color, texCoord: texCoord)

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
proc toVulY*[T](proj :GMat4[T]) :GMat4[T]=  proj*vulMat4Y
