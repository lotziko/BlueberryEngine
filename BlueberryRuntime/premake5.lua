project "BlueberryRuntime"
	kind "WindowedApp"
	language "C++"
	cppdialect "C++17"
	systemversion "latest"
	require("vendor/premake/cuda/premake5-cuda")

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")
	
	files
	{
		"src/**.h",
		"src/**.cpp",
	}

	includedirs
	{
		"src",
		"%{wks.location}/BlueberryEngine/include",
	}
	
	links
	{
		"BlueberryEngine",
	}
	
	postbuildcommands
	{
		"{COPYFILE} %{wks.location}/BlueberryEngine/vendor/hbao/lib/GFSDK_SSAO_D3D11.win64.dll %{cfg.targetdir}/GFSDK_SSAO_D3D11.win64.dll",
	}

	filter "system:windows"

	filter "configurations:Debug"
		defines "BB_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"