project "ImGui"
	kind "StaticLib"
    systemversion "latest"
	language "C++"
    cppdialect "C++17"

    targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

   	files
	{
		"imconfig.h",
		"imgui.h",
		"imgui.cpp",
		"imgui_draw.cpp",

        "imguizmo.h",
        "imguizmo.cpp",

		"imgui_tables.cpp",
		"backends/imgui_impl_win32.h",
		"backends/imgui_impl_win32.cpp",
		"backends/imgui_impl_dx11.h",
		"backends/imgui_impl_dx11.cpp",

        "misc/freetype/imgui_freetype.h",
        "misc/freetype/imgui_freetype.cpp",

        "misc/cpp/imgui_stdlib.h",
        "misc/cpp/imgui_stdlib.cpp",

		"imgui_internal.h",
		"imgui_widgets.cpp",
		"imstb_rectpack.h",
		"imstb_textedit.h",
		"imstb_truetype.h",
		"imgui_demo.cpp"
	}

    includedirs
	{
		"freetype/include"
	}
	
	links
	{
		"freetype/lib/freetype.lib"
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
