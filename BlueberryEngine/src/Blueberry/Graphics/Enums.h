#pragma once

namespace Blueberry
{
	enum TextureFormat
	{
		None,
		R8G8B8A8_UNorm,
		D24_UNorm,
	};

	enum CullMode
	{
		CullMode_None,
		CullMode_Front
	};

	enum SurfaceType
	{
		SurfaceType_Opaque,
		SurfaceType_Transparent,
		SurfaceType_DepthTransparent
	};

	enum Topology
	{
		Unknown,
		LineList,
		LineStrip,
		TriangleList,
	};
}