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
		static bool Initialize(int width, int height, void* data);

		static void ClearColor(const Color& color);
		static void SwapBuffers();

		static void SetViewport(int x, int y, int width, int height);
		static void ResizeBackbuffer(int width, int height);

		static bool CreateShader(void* vertexData, void* pixelData, GfxShader*& shader);
		static bool CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer);
		static bool CreateIndexBuffer(const UINT& indexCount, GfxIndexBuffer*& buffer);
		static bool CreateConstantBuffer(const UINT& byteSize, GfxConstantBuffer*& buffer);
		static bool CreateTexture(const TextureProperties& properties, GfxTexture*& texture);
		static bool CreateImGuiRenderer(ImGuiRenderer*& renderer);

		static void Copy(GfxTexture* source, GfxTexture* target, const Rectangle& area);

		static void SetRenderTarget(GfxTexture* renderTexture);
		static void SetGlobalConstantBuffer(const std::size_t& id, GfxConstantBuffer* buffer);
		static void SetGlobalTexture(const std::size_t& id, GfxTexture* texture);
		static void Draw(const GfxDrawingOperation& operation);

		static Matrix GetGPUMatrix(const Matrix& viewProjection);

	protected:
		virtual bool InitializeImpl(int width, int height, void* data) = 0;

		virtual void ClearColorImpl(const Color& color) const = 0;
		virtual void SwapBuffersImpl() const = 0;

		virtual void SetViewportImpl(int x, int y, int width, int height) = 0;
		virtual void ResizeBackbufferImpl(int width, int height) = 0;

		virtual bool CreateShaderImpl(void* vertexData, void* pixelData, GfxShader*& shader) = 0;
		virtual bool CreateVertexBufferImpl(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer) = 0;
		virtual bool CreateIndexBufferImpl(const UINT& indexCount, GfxIndexBuffer*& buffer) = 0;
		virtual bool CreateConstantBufferImpl(const UINT& byteSize, GfxConstantBuffer*& buffer) = 0;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) const = 0;
		virtual bool CreateImGuiRendererImpl(ImGuiRenderer*& renderer) const = 0;
		
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) const = 0;

		virtual void SetRenderTargetImpl(GfxTexture* renderTexture) = 0;
		virtual void SetGlobalConstantBufferImpl(const std::size_t& id, GfxConstantBuffer* buffer) = 0;
		virtual void SetGlobalTextureImpl(const std::size_t& id, GfxTexture* texture) = 0;
		virtual void DrawImpl(const GfxDrawingOperation& operation) const = 0;

		virtual Matrix GetGPUMatrixImpl(const Matrix& viewProjection) const = 0;

	private:
		static inline GfxDevice* s_Instance = nullptr;
	};
}