#pragma once

namespace Blueberry
{
	class VertexBuffer
	{
	public:
		virtual ~VertexBuffer() = default;
		virtual void Bind() = 0;
		virtual void SetData(float* data, const UINT& vertexCount) = 0;
	};

	class IndexBuffer
	{
	public:
		virtual ~IndexBuffer() = default;
		virtual void Bind() = 0;
		virtual void SetData(UINT* data, const UINT& indexCount) = 0;
	};

	class ConstantBuffer
	{
	public:
		virtual ~ConstantBuffer() = default;
		virtual void Bind() = 0;
		virtual void SetData(char* data, const UINT& byteCount) = 0;
	};
}