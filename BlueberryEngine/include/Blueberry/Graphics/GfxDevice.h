#pragma once

#include "GfxDrawingOperation.h"
#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	class GfxBuffer;
	class GfxTexture;
	class GfxBottomLevelAccelerationStructure;
	class GfxTopLevelAccelerationStructure;
	class GfxVertexShader;
	class GfxGeometryShader;
	class GfxFragmentShader;
	class GfxComputeShader;
	class GfxRayTracingShader;
	class ComputeShader;
	class RayTracingShader;
	class ImGuiRenderer;
	class HBAORenderer;

	class GfxDevice
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual ~GfxDevice() = default;

		static bool Initialize(int width, int height, void* data);
		static void Shutdown();

		static void ClearColor(const Color& color);
		static void ClearDepth(float depth);
		static void WaitForFrame();
		static void SwapBuffers();

		static void SetViewport(int x, int y, int width, int height);
		static void SetScissorRect(int x, int y, int width, int height);
		static void ResizeBackbuffer(int width, int height);

		static uint32_t GetViewCount();
		static void SetViewCount(uint32_t count);
		static void SetDepthBias(uint32_t bias, float slopeBias);

		static bool CreateVertexShader(const ByteData& vertexData, GfxVertexShader*& shader);
		static bool CreateGeometryShader(const ByteData& geometryData, GfxGeometryShader*& shader);
		static bool CreateFragmentShader(const ByteData& fragmentData, GfxFragmentShader*& shader);
		static bool CreateComputeShader(const ByteData& computeData, GfxComputeShader*& shader);
		static bool CreateRayTracingShader(const ByteData& rayTracingData, GfxRayTracingShader*& shader);
		static bool CreateBuffer(const BufferProperties& properties, GfxBuffer*& buffer);
		static bool CreateTexture(const TextureProperties& properties, GfxTexture*& texture);
		static bool CreateBottomLevelAccelerationStructure(const BottomLevelAccelerationStructureProperties& properties, GfxBottomLevelAccelerationStructure*& accelerationStructure);
		static bool CreateTopLevelAccelerationStructure(GfxTopLevelAccelerationStructure*& accelerationStructure);

		static void Copy(GfxTexture* source, GfxTexture* target);
		static void Copy(GfxTexture* source, GfxTexture* target, const Rectangle& area);
		static void Copy(GfxTexture* source, GfxTexture* target, const Vector2Int& offset, const Rectangle& area);
		static void Copy(GfxTexture* source, GfxTexture* target, uint32_t sourceSlice, uint32_t targetSlice, uint32_t mipLevel);

		static void SetRenderTarget(GfxTexture* renderTexture);
		static void SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture);
		static void SetRenderTarget(GfxTexture* renderTexture, uint32_t arraySlice, uint32_t mipLevel);
		static void SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, uint32_t arraySlice, uint32_t mipLevel);
		static void SetRenderTarget(GfxTexture** renderTextures, uint32_t renderTexturesCount, GfxTexture* depthStencilTexture);
		static void SetGlobalBuffer(size_t id, GfxBuffer* buffer);
		static void SetGlobalTexture(size_t id, GfxTexture* texture, uint32_t mip = 0);
		static void Draw(const GfxDrawingOperation& operation);

		static void Dispatch(ComputeShader* shader, uint32_t kernelIndex, uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ);
		static void DispatchRays(RayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, uint32_t width, uint32_t height, uint32_t depth);

		static Matrix GetGPUMatrix(const Matrix& matrix);

		static GfxDevice* GetInstance();

	protected:
		virtual bool InitializeImpl(int width, int height, void* data) = 0;

		virtual void ClearColorImpl(const Color& color) = 0;
		virtual void ClearDepthImpl(float depth) = 0;
		virtual void WaitForFrameImpl() = 0;
		virtual void SwapBuffersImpl() = 0;

		virtual void SetViewportImpl(int x, int y, int width, int height) = 0;
		virtual void SetScissorRectImpl(int x, int y, int width, int height) = 0;
		virtual void ResizeBackbufferImpl(int width, int height) = 0;

		virtual uint32_t GetViewCountImpl() = 0;
		virtual void SetViewCountImpl(uint32_t count) = 0;
		virtual void SetDepthBiasImpl(uint32_t depthBias, float depthSlopeBias) = 0;

		virtual bool CreateVertexShaderImpl(const ByteData& vertexData, GfxVertexShader*& shader) = 0;
		virtual bool CreateGeometryShaderImpl(const ByteData& geometryData, GfxGeometryShader*& shader) = 0;
		virtual bool CreateFragmentShaderImpl(const ByteData& fragmentData, GfxFragmentShader*& shader) = 0;
		virtual bool CreateComputeShaderImpl(const ByteData& computeData, GfxComputeShader*& shader) = 0;
		virtual bool CreateRayTracingShaderImpl(const ByteData& rayTracingData, GfxRayTracingShader*& shader) = 0;
		virtual bool CreateBufferImpl(const BufferProperties& properties, GfxBuffer*& buffer) = 0;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) = 0;
		virtual bool CreateBottomLevelAccelerationStructureImpl(const BottomLevelAccelerationStructureProperties& properties, GfxBottomLevelAccelerationStructure*& accelerationStructure) = 0;
		virtual bool CreateTopLevelAccelerationStructureImpl(GfxTopLevelAccelerationStructure*& accelerationStructure) = 0;
		
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target) = 0;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) = 0;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Vector2Int& offset, const Rectangle& area) = 0;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, uint32_t sourceSlice, uint32_t targetSlice, uint32_t mipLevel) = 0;

		virtual void SetRenderTargetImpl(GfxTexture** renderTextures, uint32_t renderTexturesCount, GfxTexture* depthStencilTexture, uint32_t arraySlice, uint32_t mipLevel) = 0;
		virtual void SetGlobalBufferImpl(size_t id, GfxBuffer* buffer) = 0;
		virtual void SetGlobalTextureImpl(size_t id, GfxTexture* texture, uint32_t mip) = 0;
		virtual void DrawImpl(const GfxDrawingOperation& operation) = 0;

		virtual void DispatchImpl(ComputeShader* shader, uint32_t kernelIndex, uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ) = 0;
		virtual void DispatchRaysImpl(RayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, uint32_t width, uint32_t height, uint32_t depth) = 0;
		
		virtual Matrix GetGPUMatrixImpl(const Matrix& matrix) const = 0;

	private:
		static GfxDevice* s_Instance;
	};
}