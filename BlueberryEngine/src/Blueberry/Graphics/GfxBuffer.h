#pragma once

namespace Blueberry
{
	class GfxVertexBuffer
	{
	public:
		virtual ~GfxVertexBuffer() = default;
		virtual void SetData(float* data, const uint32_t& vertexCount) = 0;
	};

	class GfxIndexBuffer
	{
	public:
		virtual ~GfxIndexBuffer() = default;
		virtual void SetData(uint32_t* data, const uint32_t& indexCount) = 0;
	};

	class GfxConstantBuffer
	{
	public:
		virtual ~GfxConstantBuffer() = default;
		virtual void SetData(char* data, const uint32_t& byteCount) = 0;
	};

	class GfxStructuredBuffer
	{
	public:
		virtual ~GfxStructuredBuffer() = default;
		virtual void SetData(char* data, const uint32_t& elementCount) = 0;
	};

	class GfxComputeBuffer
	{
	public:
		virtual ~GfxComputeBuffer() = default;
		virtual void GetData(char* data, const uint32_t& byteCount) = 0;
		virtual void SetData(char* data, const uint32_t& byteCount) = 0;
	};
}