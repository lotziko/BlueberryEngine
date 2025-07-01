#pragma once

#include "GfxDrawingOperation.h"
#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	class GfxBuffer;
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
		BB_OVERRIDE_NEW_DELETE

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
		static void SetDepthBias(const uint32_t& bias, const float& slopeBias);

		static bool CreateVertexShader(void* vertexData, GfxVertexShader*& shader);
		static bool CreateGeometryShader(void* geometryData, GfxGeometryShader*& shader);
		static bool CreateFragmentShader(void* fragmentData, GfxFragmentShader*& shader);
		static bool CreateComputeShader(void* computeData, GfxComputeShader*& shader);
		static bool CreateBuffer(const BufferProperties& properties, GfxBuffer*& buffer);
		static bool CreateTexture(const TextureProperties& properties, GfxTexture*& texture);

		static void Copy(GfxTexture* source, GfxTexture* target);
		static void Copy(GfxTexture* source, GfxTexture* target, const Rectangle& area);

		static void SetRenderTarget(GfxTexture* renderTexture);
		static void SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture);
		static void SetRenderTarget(GfxTexture* renderTexture, const uint32_t& slice);
		static void SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, const uint32_t& slice);
		static void SetGlobalBuffer(const size_t& id, GfxBuffer* buffer);
		static void SetGlobalTexture(const size_t& id, GfxTexture* texture);
		static void Draw(const GfxDrawingOperation& operation);

		static void Dispatch(GfxComputeShader* shader, const uint32_t& threadGroupsX, const uint32_t& threadGroupsY, const uint32_t& threadGroupsZ);

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
		virtual void SetDepthBiasImpl(const uint32_t& depthBias, const float& depthSlopeBias) = 0;

		virtual bool CreateVertexShaderImpl(void* vertexData, GfxVertexShader*& shader) = 0;
		virtual bool CreateGeometryShaderImpl(void* geometryData, GfxGeometryShader*& shader) = 0;
		virtual bool CreateFragmentShaderImpl(void* fragmentData, GfxFragmentShader*& shader) = 0;
		virtual bool CreateComputeShaderImpl(void* computeData, GfxComputeShader*& shader) = 0;
		virtual bool CreateBufferImpl(const BufferProperties& properties, GfxBuffer*& buffer) = 0;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) const = 0;
		
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target) const = 0;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) const = 0;

		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture) = 0;
		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, const uint32_t& slice) = 0;
		virtual void SetGlobalBufferImpl(const size_t& id, GfxBuffer* buffer) = 0;
		virtual void SetGlobalTextureImpl(const size_t& id, GfxTexture* texture) = 0;
		virtual void DrawImpl(const GfxDrawingOperation& operation) = 0;

		virtual void DispatchImpl(GfxComputeShader* shader, const uint32_t& threadGroupsX, const uint32_t& threadGroupsY, const uint32_t& threadGroupsZ) = 0;

		virtual Matrix GetGPUMatrixImpl(const Matrix& viewProjection) const = 0;

	private:
		static inline GfxDevice* s_Instance = nullptr;
	};
}