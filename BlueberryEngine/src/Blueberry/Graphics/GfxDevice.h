#pragma once

#include "GfxDrawingOperation.h"
#include "VertexLayout.h"
#include "Structs.h"

namespace Blueberry
{
	class GfxVertexBuffer;
	class GfxIndexBuffer;
	class GfxConstantBuffer;
	class GfxTexture;
	class GfxShader;
	class ImGuiRenderer;

	class GfxDevice
	{
	public:
		virtual bool Initialize(int width, int height, void* data) = 0;

		virtual void ClearColor(const Color& color) const = 0;
		virtual void SwapBuffers() const = 0;

		virtual void SetViewport(int x, int y, int width, int height) = 0;
		virtual void ResizeBackbuffer(int width, int height) = 0;

		virtual bool CreateShader(void* vertexData, void* pixelData, GfxShader*& shader) = 0;
		virtual bool CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer) = 0;
		virtual bool CreateIndexBuffer(const UINT& indexCount, GfxIndexBuffer*& buffer) = 0;
		virtual bool CreateConstantBuffer(const UINT& byteSize, GfxConstantBuffer*& buffer) = 0;
		virtual bool CreateTexture(const TextureProperties& properties, GfxTexture*& texture) const = 0;
		virtual bool CreateImGuiRenderer(ImGuiRenderer*& renderer) const = 0;
		
		virtual void SetRenderTarget(GfxTexture* renderTexture) = 0;
		virtual void SetGlobalConstantBuffer(const std::size_t& id, GfxConstantBuffer* buffer) = 0;
		virtual void SetGlobalTexture(const std::size_t& id, GfxTexture* texture) = 0;
		virtual void Draw(const GfxDrawingOperation& operation) const = 0;

		virtual Matrix GetGPUMatrix(const Matrix& viewProjection) const = 0;

	public:
		static GfxDevice* Create();
	};
}