project "BlueberryEngine"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	staticruntime "off"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	pchheader "bbpch.h"
	pchsource "src/bbpch.cpp"

	files
	{
		"src/**.h",
		"src/**.cpp",
		"vendor/stb/stb/**.h",
		"vendor/stb/stb/**.cpp",
	}

	includedirs
	{
		"src",
		"%{IncludeDir.imgui}",
		"%{IncludeDir.stb}",
	}

	links
	{
		"Imgui",
	}

filter "system:windows"
	systemversion "latest"

	defines
	{
	}