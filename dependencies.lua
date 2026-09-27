IncludeDir = {}
IncludeDir["imgui"] = "%{wks.location}/BlueberryEngine/vendor/imgui"
IncludeDir["imguinode"] = "%{wks.location}/BlueberryEditor/vendor/imguinode"
IncludeDir["jolt"] = "%{wks.location}/BlueberryEngine/vendor/jolt/src"
IncludeDir["mikktspace"] = "%{wks.location}/BlueberryEngine/vendor/mikktspace"
IncludeDir["openxr"] = "%{wks.location}/BlueberryEngine/vendor/openxr/include"
IncludeDir["rpmalloc"] = "%{wks.location}/BlueberryEngine/vendor/rpmalloc"
IncludeDir["fbxsdk"] = "%{wks.location}/BlueberryEditor/vendor/fbxsdk/include"
IncludeDir["directxtex"] = "%{wks.location}/BlueberryEditor/vendor/directxtex"
IncludeDir["directxmesh"] = "%{wks.location}/BlueberryEditor/vendor/directxmesh"
IncludeDir["cuda"] = "%{wks.location}/BlueberryBaking/vendor/cuda/include"
IncludeDir["optix"] = "%{wks.location}/BlueberryBaking/vendor/optix/include"
IncludeDir["xatlas"] = "%{wks.location}/BlueberryEngine/vendor/xatlas"
IncludeDir["miniaudio"] = "%{wks.location}/BlueberryEngine/vendor/miniaudio"
IncludeDir["rmlui"] = "%{wks.location}/BlueberryEngine/vendor/rmlui/include"
IncludeDir["lz4"] = "%{wks.location}/BlueberryEngine/vendor/lz4"
IncludeDir["d3dx12"] = "%{wks.location}/BlueberryEngine/vendor/d3dx12/include"
IncludeDir["dxc"] = "%{wks.location}/BlueberryEngine/vendor/dxc/include"

Library = {}
Library["openxr"] = "%{wks.location}/BlueberryEngine/vendor/openxr/native/x64/release/lib/openxr_loader.lib"
Library["fbxsdk"] = "%{wks.location}/BlueberryEditor/vendor/fbxsdk/lib/vs2017/x64/release/libfbxsdk.lib"
Library["dxc"] = "%{wks.location}/BlueberryEngine/vendor/dxc/lib/x64/dxcompiler.lib"

LibraryDir = {}
LibraryDir["cuda"] = "%{wks.location}/BlueberryBaking/vendor/cuda/lib"