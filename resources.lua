project "BlueberryResources"
	kind "Utility"
	language "C++"
	systemversion "latest"
	
	fastuptodate "Off"
	dependson { "BlueberryBaking" }

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")
	
	copychanged
	{
		{ "%{wks.location}/BlueberryEditor/assets", "%{wks.location}/bin/" .. outputdir .. "/BlueberryEditor/assets" },
		{ "%{wks.location}/BlueberryEngine/assets", "%{wks.location}/bin/" .. outputdir .. "/BlueberryEditor/assets" },
		{ "%{wks.location}/bin/" .. outputdir .. "/BlueberryBaking/assets", "%{wks.location}/bin/" .. outputdir .. "/BlueberryEditor/assets" },
		{ "%{wks.location}/BlueberryEngine/include", "%{wks.location}/bin/" .. outputdir .. "/BlueberryEditor/include" },
		{ "%{wks.location}/BlueberryEngine/vendor/rmlui/include", "%{wks.location}/bin/" .. outputdir .. "/BlueberryEditor/include" },
	}
