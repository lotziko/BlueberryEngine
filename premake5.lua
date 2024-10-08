include "dependencies.lua"

workspace "BlueberryEngine"
	architecture "x64"
	startproject "BlueberryEditor"

	configurations
	{
		"Debug",
		"Release"
	}

	flags
	{
		"MultiProcessorCompile"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

group "Dependencies"
	include "BlueberryEngine/vendor/imgui"
group ""

include "BlueberryEngine"
include "BlueberryEditor"