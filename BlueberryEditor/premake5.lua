project "BlueberryEditor"
	kind "WindowedApp"
	language "C++"
	cppdialect "C++17"
	systemversion "latest"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
		"src/**.h",
		"src/**.cpp",
		"vendor/stb/**.h",
		"vendor/stb/**.cpp",
		"vendor/rapidyaml/**.h",
		"vendor/rapidyaml/**.cpp"
	}

	includedirs
	{
		"src",
		"%{wks.location}/BlueberryEngine/src",
		"%{IncludeDir.imgui}",
		"%{IncludeDir.stb}",
		"%{IncludeDir.rapidyaml}"
	}

	links
	{
		"BlueberryEngine"
	}

	filter "system:windows"

	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"