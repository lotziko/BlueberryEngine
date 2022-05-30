project "BlueberryEditor"
	kind "WindowedApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "off"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
		"src/**.h",
		"src/**.cpp"
	}

	includedirs
	{
		"%{wks.location}/BlueberryEngine/src",
		"%{wks.location}/BlueberryEngine/vendor",
	}

	links
	{
		"BlueberryEngine",
	}

filter "system:windows"
	systemversion "latest"

	defines
	{
	}