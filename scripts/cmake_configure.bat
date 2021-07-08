cmake -B ../output/build_x64 -A x64 -DCMAKE_TOOLCHAIN_FILE=../third_libraries/vcpkg/scripts/buildsystems/vcpkg.cmake ..

::cmake -B ../output/build_vs15_x64 -G "Visual Studio 14 2015" -A x64 -DCMAKE_TOOLCHAIN_FILE=../third_libraries/vcpkg/scripts/buildsystems/vcpkg.cmake ..
::cmake -B ../output/build_vs17_x64 -G "Visual Studio 15 2017" -A x64 -DCMAKE_TOOLCHAIN_FILE=../third_libraries/vcpkg/scripts/buildsystems/vcpkg.cmake ..


timeout 20