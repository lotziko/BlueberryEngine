#pragma once

namespace Blueberry
{
	enum TextureType
	{
		Resource,
		RenderTarget,
		Staging
	};

	enum Topology
	{
		Unknown,
		LineList,
		LineStrip,
		TriangleList,
	};
}