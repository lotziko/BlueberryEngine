#pragma once

namespace Blueberry
{
	class GfxVertexBuffer
	{
	public:
		virtual ~GfxVertexBuffer() = default;
		virtual void SetData(float* data, const UINT& vertexCount) = 0;
	};

	class GfxIndexBuffer
	{
	public:
		virtual ~GfxIndexBuffer() = default;
		virtual void SetData(UINT* data, const UINT& indexCount) = 0;
	};

	class GfxConstantBuffer
	{
	public:
		virtual ~GfxConstantBuffer() = default;
		virtual void SetData(char* data, const UINT& byteCount) = 0;
	};

	class GfxComputeBuffer
	{
	public:
		virtual ~GfxComputeBuffer() = default;
		virtual void GetData(char* data, const UINT& byteCount) = 0;
		virtual void SetData(char* data, const UINT& byteCount) = 0;
	};
}