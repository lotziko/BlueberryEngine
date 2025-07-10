project "BlueberryBaking"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	systemversion "latest"
	require("vendor/premake/cuda/premake5-cuda")

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")
	cudaPtxDir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}/assets/ptx")
	
	buildcustomizations("BuildCustomizations/CUDA 11.8")
	
	cudaPTXFiles 
	{ 
		"**.cu"
	}

	files
	{
		"include/**.h",
		"src/**.h",
		"src/**.cpp",
	}

	includedirs
	{
		"include",
		"%{wks.location}/BlueberryEngine/include",
		"%{IncludeDir.cuda}",
		"%{IncludeDir.optix}",
	}

	libdirs
	{
		"%{LibraryDir.cuda}",
	}

	links
	{
		"cuda.lib",
		"cudadevrt.lib",
		"cudart.lib",
	}

	cudaFastMath "On"

	filter "system:windows"

	filter "configurations:Debug"
		defines "BB_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"