RD /S /Q "..\output"
RD /S /Q "..\.vs"

DEL /S /Q "..\*.sln"
DEL /S /Q /AH "..\*.sln"
DEL /S /Q "..\*.sdf"
DEL /S /Q /AH "..\*.sdf"
DEL /S /Q "..\*.suo"
DEL /S /Q /AH "..\*.suo"

set subdir = ""
DEL /S /Q "..\%subdir%\*.vcxproj"
DEL /S /Q /AH "..\%subdir%\*.vcxproj"
DEL /S /Q "..\%subdir%\*.filters"
DEL /S /Q /AH "..\%subdir%\*.filters"
DEL /S /Q "..\%subdir%\*.user"
DEL /S /Q /AH "..\%subdir%\*.user"

pause
