#pragma once

namespace Blueberry
{
	enum class TextureFormat
	{
		None,
		R8G8B8A8_UNorm,
		D24_UNorm,
	};

	enum class CullMode
	{
		None,
		Front,
		Back
	};

	enum class SurfaceType
	{
		Opaque,
		Transparent,
		DepthTransparent
	};

	enum class Topology
	{
		Unknown,
		LineList,
		LineStrip,
		TriangleList,
	};
}