project "ImGuiNode"
	kind "StaticLib"
    systemversion "latest"
	language "C++"
    cppdialect "C++17"

    targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

   	files
	{
		"imgui_node_editor.h",
		"imgui_node_editor.cpp",
		"imgui_node_editor_api.cpp",
		"imgui_node_editor_internal.h",
		"imgui_node_editor_internal.inl",
		"imgui_extra_math.h",
		"imgui_extra_math.inl",
		"imgui_bezier_math.h",
		"imgui_bezier_math.inl",
		"imgui_canvas.h",
		"imgui_canvas.cpp",
		"crude_json.h",
		"crude_json.cpp",
	}

    includedirs
	{
		"%{IncludeDir.imgui}",
	}
	
	links
	{
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