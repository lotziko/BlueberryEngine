project "RmlUi"
	kind "StaticLib"
    systemversion "latest"
	language "C++"
    cppdialect "C++17"

    targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

   	files
	{
		"Include/**.h",
		"Dependencies/**.h",
		"Source/**.h",
		"Source/**.cpp",
	}

    includedirs
	{
		"Dependencies/include"
	}
	
	links
	{
		"Dependencies/lib/freetype.lib"
	}

	defines { "WIN32", "_WINDOWS", "NDEBUG", "_CRT_SECURE_NO_WARNINGS", "RMLUI_VERSION=\"6.2\"", "RMLUI_FONT_ENGINE_FREETYPE", "RMLUI_CORE_EXPORTS", "RMLUI_DEBUGGER_EXPORTS" }

    filter "system:windows"

    filter "system:linux"
		pic "On"

	filter "configurations:Debug"
        staticruntime "off"
		runtime "Debug"

	filter "configurations:Release"
        staticruntime "off"
		runtime "Release"
		optimize "size"