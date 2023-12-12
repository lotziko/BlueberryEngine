project "BlueberryEngine"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	systemversion "latest"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	pchheader "bbpch.h"
	pchsource "src/bbpch.cpp"

	files
	{
		"src/**.h",
		"src/**.cpp"
	}

	includedirs
	{
		"src",
		"%{IncludeDir.imgui}"
	}

	links
	{
		"Imgui"
	}

	filter "system:windows"

	filter "system:linux"
		pic "On"

	filter "configurations:Debug"
		staticruntime "off"
		runtime "Debug"

	filter "configurations:Release"
		staticruntime "off"
		runtime "Release"