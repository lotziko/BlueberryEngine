#pragma once

#include "GfxDrawingOperation.h"
#include "VertexLayout.h"
#include "Structs.h"

namespace Blueberry
{
	class GfxVertexBuffer;
	class GfxIndexBuffer;
	class GfxConstantBuffer;
	class GfxStructuredBuffer;
	class GfxComputeBuffer;
	class GfxTexture;
	class GfxVertexShader;
	class GfxGeometryShader;
	class GfxFragmentShader;
	class GfxComputeShader;
	class ImGuiRenderer;
	class HBAORenderer;

	class GfxDevice
	{
	public:
		static bool Initialize(int width, int height, void* data);
		static void Shutdown();

		static void ClearColor(const Color& color);
		static void ClearDepth(const float& depth);
		static void SwapBuffers();

		static void SetViewport(int x, int y, int width, int height);
		static void SetScissorRect(int x, int y, int width, int height);
		static void ResizeBackbuffer(int width, int height);

		static const uint32_t& GetViewCount();
		static void SetViewCount(const uint32_t& count);

		static bool CreateVertexShader(void* vertexData, GfxVertexShader*& shader);
		static bool CreateGeometryShader(void* geometryData, GfxGeometryShader*& shader);
		static bool CreateFragmentShader(void* fragmentData, GfxFragmentShader*& shader);
		static bool CreateComputeShader(void* computeData, GfxComputeShader*& shader);
		static bool CreateVertexBuffer(const VertexLayout& layout, const uint32_t& vertexCount, GfxVertexBuffer*& buffer);
		static bool CreateIndexBuffer(const uint32_t& indexCount, GfxIndexBuffer*& buffer);
		static bool CreateConstantBuffer(const uint32_t& byteCount, GfxConstantBuffer*& buffer);
		static bool CreateStructuredBuffer(const uint32_t& elementCount, const uint32_t& elementSize, GfxStructuredBuffer*& buffer);
		static bool CreateComputeBuffer(const uint32_t& elementCount, const uint32_t& elementSize, GfxComputeBuffer*& buffer);
		static bool CreateTexture(const TextureProperties& properties, GfxTexture*& texture);

		static void Copy(GfxTexture* source, GfxTexture* target);
		static void Copy(GfxTexture* source, GfxTexture* target, const Rectangle& area);
		static void Read(GfxTexture* source, void* target);
		static void Read(GfxTexture* source, void* target, const Rectangle& area);

		static void SetRenderTarget(GfxTexture* renderTexture);
		static void SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture);
		static void SetRenderTarget(GfxTexture* renderTexture, const uint32_t& slice);
		static void SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, const uint32_t& slice);
		static void SetGlobalConstantBuffer(const size_t& id, GfxConstantBuffer* buffer);
		static void SetGlobalStructuredBuffer(const size_t& id, GfxStructuredBuffer* buffer);
		static void SetGlobalTexture(const size_t& id, GfxTexture* texture);
		static void Draw(const GfxDrawingOperation& operation);

		static void Dispatch(GfxComputeShader*& shader, const uint32_t& threadGroupsX, const uint32_t& threadGroupsY, const uint32_t& threadGroupsZ);

		static Matrix GetGPUMatrix(const Matrix& viewProjection);

		static GfxDevice* GetInstance();

	protected:
		virtual bool InitializeImpl(int width, int height, void* data) = 0;

		virtual void ClearColorImpl(const Color& color) const = 0;
		virtual void ClearDepthImpl(const float& depth) const = 0;
		virtual void SwapBuffersImpl() = 0;

		virtual void SetViewportImpl(int x, int y, int width, int height) = 0;
		virtual void SetScissorRectImpl(int x, int y, int width, int height) = 0;
		virtual void ResizeBackbufferImpl(int width, int height) = 0;

		virtual const uint32_t& GetViewCountImpl() = 0;
		virtual void SetViewCountImpl(const uint32_t& count) = 0;

		virtual bool CreateVertexShaderImpl(void* vertexData, GfxVertexShader*& shader) = 0;
		virtual bool CreateGeometryShaderImpl(void* geometryData, GfxGeometryShader*& shader) = 0;
		virtual bool CreateFragmentShaderImpl(void* fragmentData, GfxFragmentShader*& shader) = 0;
		virtual bool CreateComputeShaderImpl(void* computeData, GfxComputeShader*& shader) = 0;
		virtual bool CreateVertexBufferImpl(const VertexLayout& layout, const uint32_t& vertexCount, GfxVertexBuffer*& buffer) = 0;
		virtual bool CreateIndexBufferImpl(const uint32_t& indexCount, GfxIndexBuffer*& buffer) = 0;
		virtual bool CreateConstantBufferImpl(const uint32_t& byteCount, GfxConstantBuffer*& buffer) = 0;
		virtual bool CreateStructuredBufferImpl(const uint32_t& elementCount, const uint32_t& elementSize, GfxStructuredBuffer*& buffer) = 0;
		virtual bool CreateComputeBufferImpl(const uint32_t& elementCount, const uint32_t& elementSize, GfxComputeBuffer*& buffer) = 0;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) const = 0;
		
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target) const = 0;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) const = 0;
		virtual void ReadImpl(GfxTexture* source, void* target) const = 0;
		virtual void ReadImpl(GfxTexture* source, void* target, const Rectangle& area) const = 0;

		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture) = 0;
		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, const uint32_t& slice) = 0;
		virtual void SetGlobalConstantBufferImpl(const size_t& id, GfxConstantBuffer* buffer) = 0;
		virtual void SetGlobalStructuredBufferImpl(const size_t& id, GfxStructuredBuffer* buffer) = 0;
		virtual void SetGlobalTextureImpl(const size_t& id, GfxTexture* texture) = 0;
		virtual void DrawImpl(const GfxDrawingOperation& operation) = 0;

		virtual void DispatchImpl(GfxComputeShader*& shader, const uint32_t& threadGroupsX, const uint32_t& threadGroupsY, const uint32_t& threadGroupsZ) const = 0;

		virtual Matrix GetGPUMatrixImpl(const Matrix& viewProjection) const = 0;

	private:
		static inline GfxDevice* s_Instance = nullptr;
	};
}