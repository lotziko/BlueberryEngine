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
		Point,

		CompareDepth
	};

	enum class TextureFormat
	{
		None = 0,
		R16G16B16A16_Float = 10,
		R16G16B16A16_UNorm = 11,
		R8G8B8A8_UNorm = 28,
		R8G8B8A8_UNorm_SRGB = 29,
		R8G8B8A8_UInt = 30,
		R24G8_Typeless = 44,
		D24_UNorm = 45,
		D32_Float = 40,
		BC1_UNorm = 71,
		BC2_UNorm = 74,
		BC3_UNorm = 77,
		BC3_UNorm_SRGB = 78,
		BC5_UNorm = 83,
		BC6H_UFloat = 95,
		BC7_UNorm = 98,
		BC7_UNorm_SRGB = 99,
	};

	enum class BufferType
	{
		Vertex,
		Index,
		Structured,
		Raw,
		Constant
	};

	enum class BufferFormat
	{
		R32_Float = 41,
		R32_UInt = 42,
	};

	enum class TextureDimension
	{
		Texture2D,
		Texture2DArray,
		TextureCube,
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

	enum class ZTest
	{
		Never,
		Less,
		Equal,
		LessEqual,
		Greater,
		NotEqual,
		GreaterEqual,
		Always
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
		PointList,
		LineList,
		LineStrip,
		TriangleList
	};
}