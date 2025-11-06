project "BlueberryEngine"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	systemversion "latest"

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

	--pchheader "bbpch.h"
	--pchsource "src/bbpch.cpp"

	files
	{
		"include/**.h",
		"src/**.h",
		"src/**.cpp",
		"vendor/mikktspace/**.h",
		"vendor/mikktspace/**.cpp",
		"vendor/hbao/include/**.h",
		"vendor/openxr/include/**.h",
		"vendor/rpmalloc/**.h",
		"vendor/rpmalloc/**.cpp",
		"vendor/xatlas/**.h",
		"vendor/xatlas/**.cpp",
	}

	includedirs
	{
		"include",
		"%{IncludeDir.imgui}",
		"%{IncludeDir.jolt}",
		"%{IncludeDir.mikktspace}",
		"%{IncludeDir.hbao}",
		"%{IncludeDir.openxr}",
		"%{IncludeDir.flathashmap}",
		"%{IncludeDir.rpmalloc}",
	}

	links
	{
		"Imgui",
		"Jolt",
		"%{Library.hbao}",
		"%{Library.openxr}",
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
		optimize "on"