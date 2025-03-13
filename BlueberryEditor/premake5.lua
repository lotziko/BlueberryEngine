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
		"vendor/rapidyaml/**.cpp",
		"vendor/fbxsdk/include/**.h",
		"vendor/directxtex/**.h",
		"vendor/directxtex/**.cpp",
	}

	includedirs
	{
		"src",
		"%{wks.location}/BlueberryEngine/src",
		"%{IncludeDir.imgui}",
		"%{IncludeDir.stb}",
		"%{IncludeDir.rapidyaml}",
		"%{IncludeDir.fbxsdk}",
		"%{IncludeDir.directxtex}",
		"%{IncludeDir.flathashmap}",
	}

	links
	{
		"BlueberryEngine",
		"%{LibraryDir.fbxsdk}",
	}

	postbuildcommands
	{
		"{COPYDIR} %{wks.location}/BlueberryEditor/assets %{cfg.targetdir}/assets",
		"{COPYDIR} %{wks.location}/BlueberryEngine/assets %{cfg.targetdir}/assets",
		"{COPYFILE} %{wks.location}/BlueberryEditor/vendor/fbxsdk/lib/vs2017/x64/release/libfbxsdk.dll %{cfg.targetdir}/libfbxsdk.dll",
		"{COPYFILE} %{wks.location}/BlueberryEngine/vendor/hbao/lib/GFSDK_SSAO_D3D11.win64.dll %{cfg.targetdir}/GFSDK_SSAO_D3D11.win64.dll",
		"{COPYFILE} %{wks.location}/BlueberryEngine/vendor/openxr/native/x64/release/bin/openxr_loader.dll %{cfg.targetdir}/openxr_loader.dll",
	}

	filter "system:windows"

	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"