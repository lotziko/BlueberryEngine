#pragma once

namespace Blueberry
{
	enum class WrapMode
	{
		Repeat,
		Clamp
	};

	enum class FilterMode
	{
		Linear,
		Point
	};

	enum class TextureFormat
	{
		None,
		R8G8B8A8_UNorm,
		R8G8B8A8_UNorm_SRGB,
		R16G16B16A16_FLOAT,
		R8G8B8A8_UINT,
		D24_UNorm,
	};

	enum class CullMode
	{
		None,
		Front,
		Back
	};

	enum class BlendMode
	{
		One,
		Zero,
		SrcAlpha,
		OneMinusSrcAlpha
	};

	enum class ZWrite
	{
		Off,
		On
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