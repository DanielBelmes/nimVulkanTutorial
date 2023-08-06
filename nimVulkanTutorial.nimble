# Package

version       = "0.1.0"
author        = "DanielBelmes"
description   = "Vulkan Tutorial"
license       = "MIT"
srcDir        = "src"
binDir        = "build"
backend       = "c"
bin           = @["nimVulkanTutorial"]


# Dependencies

requires "https://github.com/DanielBelmes/glfw#head"
requires "https://github.com/DanielBelmes/vulkan#head"
requires "https://github.com/DanielBelmes/stb_nim#head"
requires "https://github.com/DanielBelmes/ObjLoader#head"
requires "vmath"

before build:
  exec("glslc src/shaders/shader.vert -o src/shaders/vert.spv")
  exec("glslc src/shaders/shader.frag -o src/shaders/frag.spv")

task clean, "Cleans binaries":
  echo "❯ Removing Build Dir"
  exec("rm -rf ./build")