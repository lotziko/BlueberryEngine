project "Jolt"
	kind "StaticLib"
    systemversion "latest"
	language "C++"
    cppdialect "C++17"

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

   	files
	{
		"src/**.h",
		"src/**.cpp",
	}

	includedirs
	{
		"src"
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
