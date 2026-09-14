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
		"{COPYFILE} %{wks.location}/BlueberryEngine/vendor/dxc/bin/x64/dxcompiler.dll %{cfg.targetdir}/dxcompiler.dll",
		"{COPYFILE} %{wks.location}/BlueberryEngine/vendor/d3dx12/bin/x64/D3D12Core.dll %{cfg.targetdir}/D3D12Core.dll",
		"{COPYFILE} %{wks.location}/BlueberryEngine/vendor/d3dx12/bin/x64/d3d12SDKLayers.dll %{cfg.targetdir}/d3d12SDKLayers.dll",
	}

	filter "system:windows"

	filter "configurations:Debug"
		defines "BB_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"