#pragma once

namespace Blueberry
{
	enum TextureType
	{
		Resource,
		RenderTarget,
		DepthStencil,
		Staging
	};

	enum TextureFormat
	{
		None,
		R8G8B8A8_UNorm,
		D24_UNorm,
	};

	enum SurfaceType
	{
		Opaque,
		Transparent,
		DepthTransparent
	};

	enum Topology
	{
		Unknown,
		LineList,
		LineStrip,
		TriangleList,
	};
}