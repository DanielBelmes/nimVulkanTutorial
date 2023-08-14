import application

if isMainModule:
  var app: VulkanTutorialApp

  try:
    app.run()
  except CatchableError:
    echo getCurrentExceptionMsg()
    quit(-1)