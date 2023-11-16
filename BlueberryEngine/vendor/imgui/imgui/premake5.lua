project "ImGui"
	kind "StaticLib"
    systemversion "latest"
	language "C++"
    cppdialect "C++17"

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

   	files
	{
		"imconfig.h",
		"imgui.h",
		"imgui.cpp",
		"imgui_draw.cpp",

		"imgui_tables.cpp",
		"backends/imgui_impl_win32.h",
		"backends/imgui_impl_win32.cpp",
		"backends/imgui_impl_dx11.h",
		"backends/imgui_impl_dx11.cpp",

		"imgui_internal.h",
		"imgui_widgets.cpp",
		"imstb_rectpack.h",
		"imstb_textedit.h",
		"imstb_truetype.h",
		"imgui_demo.cpp"
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
