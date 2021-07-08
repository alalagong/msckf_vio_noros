workspace "msckf_sln"
	cur_dir = os.getcwd()

	location (cur_dir)
	configurations { "Release", "Debug" }
	platforms { "x64" }
	objdir (cur_dir .. "/output/build")
	targetdir (cur_dir .. "/output/bin/%{cfg.buildcfg}")
	buildoptions { "/bigobj" }
	
	using_viz = false
	if using_viz then
		defines { "USING_VIZ" }
	end
	
	defines { "PROJECT_DIR=" .. "\"" .. cur_dir .. "\"" }
	
	filter "configurations:Release"
		defines { "NDEBUGS", "EIGEN_NO_DEBUG", "EIGEN_NO_DEBUG" }
		symbols "On"
		optimize "Speed"
		runtime "Release"
	filter {}
	filter "configurations:Debug"
		defines { "DEBUGS", "_DEBUG", "EIGEN_NO_DEBUG", "EIGEN_NO_DEBUG" }
		symbols "On"
		optimize "Off"
		runtime "Debug"
	filter {}
	
	filter "configurations:*32"
		architecture "x86"
	filter {}
	filter "configurations:*64"
		architecture "x86_64"
	filter {}
	
	-- add OpenCV library
	if os.target() == "windows" then
		OpenCV_dir = cur_dir .. "/third_libraries/OpenCV"
		OpenCV_inc_dir = OpenCV_dir .. "/include"
		if _ACTION == "vs2013" then
			OpenCV_lib_dir = OpenCV_dir .. "/lib/x64_vc12"
		else
			OpenCV_lib_dir = OpenCV_dir .. "/lib/x64_vc14"
		end
		
		includedirs { OpenCV_inc_dir }
		libdirs { OpenCV_lib_dir }
			
		filter "configurations:Release"
			links { "opencv_world310" }
			if using_viz then
				links { "opencv_viz310" }
			end
		filter {}

		filter "configurations:Debug"
			links { "opencv_world310d" }
			if using_viz then
				links { "opencv_viz310d" }
			end
		filter {}
	elseif os.target() == "linux" then
		includedirs { "/usr/local/include" }
		libdirs { "/usr/local/lib" }
		links { "opencv_core", "opencv_highgui", "opencv_imgproc" }
		if using_viz then
			links { "opencv_viz" }
		end
	end
		
	-- add Eigen library
	Eigen_dir = cur_dir .. "/third_libraries/Eigen/eigen-3.3.7"
	includedirs { Eigen_dir }
		
	-- add environment variables
	if os.target() == "windows" then
		if _ACTION == "vs2013" then
			OpenCV_env = OpenCV_dir .. "/bin/vc12;"
		else
			OpenCV_env = OpenCV_dir .. "/bin/vc14;"
		end
		env_vars = "PATH=%PATH%;" .. OpenCV_env
		debugenvs(env_vars)
	end
		
project "msckf_pro"
	kind "ConsoleApp"
	language "C++"
	location (cur_dir .. "/msckf_vio")
	
	includedirs {
		"./msckf_vio/include",
		"./msckf_vio/core",
		"./msckf_vio/system",
		"./msckf_vio/util",
	}
	
	files {
		"./msckf_vio/include/msckf_vio/**",
		"./msckf_vio/core/**",
		"./msckf_vio/util/**",
		"./msckf_vio/system/**",
		"./msckf_vio/entry/**"
	}
