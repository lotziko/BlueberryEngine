project "BlueberryEditor"
	kind "WindowedApp"
	language "C++"
	cppdialect "C++17"
	systemversion "latest"

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
		"src/**.h",
		"src/**.cpp",
		"vendor/rapidyaml/**.h",
		"vendor/rapidyaml/**.cpp",
		"vendor/fbxsdk/include/**.h",
		"vendor/directxtex/**.h",
		"vendor/directxtex/**.cpp",
		"vendor/xatlas/**.h",
		"vendor/xatlas/**.cpp",
	}

	includedirs
	{
		"src",
		"%{wks.location}/BlueberryEngine/include",
		"%{wks.location}/BlueberryBaking/include",
		"%{IncludeDir.imgui}",
		"%{IncludeDir.rapidyaml}",
		"%{IncludeDir.fbxsdk}",
		"%{IncludeDir.directxtex}",
		"%{IncludeDir.flathashmap}",
		"%{IncludeDir.xatlas}",
	}
	
	links
	{
		"BlueberryEngine",
		"BlueberryBaking",
		"%{Library.fbxsdk}",
	}

	postbuildcommands
	{
		"{COPYDIR} %{wks.location}/BlueberryEditor/assets %{cfg.targetdir}/assets",
		"{COPYDIR} %{wks.location}/BlueberryEngine/assets %{cfg.targetdir}/assets",
		"{COPYDIR} %{wks.location}/bin/" .. outputdir .. "/BlueberryBaking/assets %{cfg.targetdir}/assets",
		"{COPYDIR} %{wks.location}/BlueberryEngine/include %{cfg.targetdir}/include",
		"{COPYFILE} %{wks.location}/BlueberryEditor/vendor/fbxsdk/lib/vs2017/x64/release/libfbxsdk.dll %{cfg.targetdir}/libfbxsdk.dll",
		"{COPYFILE} %{wks.location}/BlueberryEngine/vendor/hbao/lib/GFSDK_SSAO_D3D11.win64.dll %{cfg.targetdir}/GFSDK_SSAO_D3D11.win64.dll",
		"{COPYFILE} %{wks.location}/BlueberryEngine/vendor/openxr/native/x64/release/bin/openxr_loader.dll %{cfg.targetdir}/openxr_loader.dll",
		"{COPYFILE} %{wks.location}/bin/" .. outputdir .. "/BlueberryEngine/BlueberryEngine.lib %{cfg.targetdir}/BlueberryEngine.lib",
		"{COPYFILE} %{wks.location}/BlueberryEditor/vendor/fastbuild/bin/FBuild.exe %{cfg.targetdir}/FBuild.exe",
	}

	filter "system:windows"

	filter "configurations:Debug"
		defines "BB_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"