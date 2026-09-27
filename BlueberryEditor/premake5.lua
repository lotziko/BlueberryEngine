project "BlueberryEditor"
	kind "WindowedApp"
	language "C++"
	cppdialect "C++17"
	systemversion "latest"

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
		"assets/**",
		"src/**.h",
		"src/**.cpp",
		"vendor/fbxsdk/include/**.h",
		"vendor/directxtex/**.h",
		"vendor/directxtex/**.cpp",
		"vendor/directxmesh/**.h",
		"vendor/directxmesh/**.cpp",
	}

	includedirs
	{
		"src",
		"%{wks.location}/BlueberryEngine/include",
		"%{wks.location}/BlueberryBaking/include",
		"%{IncludeDir.jolt}",
		"%{IncludeDir.imgui}",
		"%{IncludeDir.imguinode}",
		"%{IncludeDir.rapidyaml}",
		"%{IncludeDir.fbxsdk}",
		"%{IncludeDir.directxtex}",
		"%{IncludeDir.directxmesh}",
		"%{IncludeDir.flathashmap}",
		"%{IncludeDir.xatlas}",
		"%{IncludeDir.dxc}",
	}
	
	links
	{
		"BlueberryEngine",
		"BlueberryBaking",
		"%{Library.fbxsdk}",
		"ImguiNode",
	}

	dependson { "BlueberryResources", "BlueberryRuntime" }

	copychanged
	{
		{ "%{wks.location}/BlueberryEditor/vendor/fbxsdk/lib/vs2017/x64/release/libfbxsdk.dll", "%{cfg.targetdir}/libfbxsdk.dll" },
		{ "%{wks.location}/BlueberryEngine/vendor/openxr/native/x64/release/bin/openxr_loader.dll", "%{cfg.targetdir}/openxr_loader.dll" },
		{ "%{wks.location}/BlueberryEngine/vendor/dxc/bin/x64/dxcompiler.dll", "%{cfg.targetdir}/dxcompiler.dll" },
		{ "%{wks.location}/BlueberryEngine/vendor/dxc/bin/x64/dxil.dll", "%{cfg.targetdir}/dxil.dll" },
		{ "%{wks.location}/BlueberryEngine/vendor/d3dx12/bin/x64/D3D12Core.dll", "%{cfg.targetdir}/D3D12Core.dll" },
		{ "%{wks.location}/BlueberryEngine/vendor/d3dx12/bin/x64/d3d12SDKLayers.dll", "%{cfg.targetdir}/d3d12SDKLayers.dll" },
		{ "%{wks.location}/bin/" .. outputdir .. "/BlueberryRuntime/BlueberryRuntime.lib", "%{cfg.targetdir}/BlueberryRuntime.lib" },
		{ "%{wks.location}/bin/" .. outputdir .. "/BlueberryRuntime/BlueberryRuntime.exe", "%{cfg.targetdir}/BlueberryRuntime.exe" },
		{ "%{wks.location}/bin/" .. outputdir .. "/RmlUi/RmlUi.lib", "%{cfg.targetdir}/rmlui.lib" },
		{ "%{wks.location}/BlueberryEditor/vendor/fastbuild/bin/FBuild.exe", "%{cfg.targetdir}/FBuild.exe" },
		{ "%{wks.location}/vendor/premake/bin/premake5.exe", "%{cfg.targetdir}/premake5.exe" },
	}

	filter "files:assets/**"
		buildaction "None"
	filter {}

	filter "system:windows"

	filter "configurations:Debug"
		defines "BB_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"