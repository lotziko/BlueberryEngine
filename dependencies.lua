IncludeDir = {}
IncludeDir["imgui"] = "%{wks.location}/BlueberryEngine/vendor/imgui"
IncludeDir["jolt"] = "%{wks.location}/BlueberryEngine/vendor/jolt/src"
IncludeDir["mikktspace"] = "%{wks.location}/BlueberryEngine/vendor/mikktspace"
IncludeDir["hbao"] = "%{wks.location}/BlueberryEngine/vendor/hbao/include"
IncludeDir["openxr"] = "%{wks.location}/BlueberryEngine/vendor/openxr/include"
IncludeDir["flathashmap"] = "%{wks.location}/BlueberryEngine/vendor/flathashmap"
IncludeDir["rpmalloc"] = "%{wks.location}/BlueberryEngine/vendor/rpmalloc"
IncludeDir["rapidyaml"] = "%{wks.location}/BlueberryEditor/vendor/rapidyaml"
IncludeDir["fbxsdk"] = "%{wks.location}/BlueberryEditor/vendor/fbxsdk/include"
IncludeDir["directxtex"] = "%{wks.location}/BlueberryEditor/vendor/directxtex"
IncludeDir["cuda"] = "%{wks.location}/BlueberryBaking/vendor/cuda/include"
IncludeDir["optix"] = "%{wks.location}/BlueberryBaking/vendor/optix/include"
IncludeDir["xatlas"] = "%{wks.location}/BlueberryEngine/vendor/xatlas"

Library = {}
Library["hbao"] = "%{wks.location}/BlueberryEngine/vendor/hbao/lib/GFSDK_SSAO_D3D11.win64.lib"
Library["openxr"] = "%{wks.location}/BlueberryEngine/vendor/openxr/native/x64/release/lib/openxr_loader.lib"
Library["fbxsdk"] = "%{wks.location}/BlueberryEditor/vendor/fbxsdk/lib/vs2017/x64/release/libfbxsdk.lib"

LibraryDir = {}
LibraryDir["cuda"] = "%{wks.location}/BlueberryBaking/vendor/cuda/lib"