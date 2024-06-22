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
		None = 0,
		R16G16B16A16_FLOAT = 10,
		R8G8B8A8_UNorm = 28,
		R8G8B8A8_UNorm_SRGB = 29,
		R8G8B8A8_UINT = 30,
		R24G8_TYPELESS = 44,
		D24_UNorm = 45,
		BC3_UNORM = 77,
		BC3_UNORM_SRGB = 78,
		BC7_UNORM = 98,
		BC7_UNORM_SRGB = 99,
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