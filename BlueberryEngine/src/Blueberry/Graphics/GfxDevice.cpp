#include "Blueberry\Graphics\GfxDevice.h"

#include "Blueberry\Graphics\GraphicsAPI.h"
#include "Blueberry\Graphics\Mesh.h"

#include "..\..\Concrete\DX11\GfxDeviceDX11.h"
#include "..\..\Concrete\DX12\GfxDeviceDX12.h"

namespace Blueberry
{
	GfxDevice* GfxDevice::s_Instance = nullptr;

	bool GfxDevice::Initialize(int width, int height, void* data)
	{
		switch (GraphicsAPI::GetAPI())
		{
		case GraphicsAPI::API::None:
			BB_ERROR("API doesn't exist.");
			return false;
		case GraphicsAPI::API::DX11:
			s_Instance = new GfxDeviceDX11();
			break;
		case GraphicsAPI::API::DX12:
			s_Instance = new GfxDeviceDX12();
			break;
		}
		
		return s_Instance->InitializeImpl(width, height, data);
	}

	void GfxDevice::Shutdown()
	{
		if (s_Instance != nullptr)
		{
			delete s_Instance;
			s_Instance = nullptr;
		}
	}

	void GfxDevice::ClearColor(const Color& color)
	{
		s_Instance->ClearColorImpl(color);
	}

	void GfxDevice::ClearDepth(float depth)
	{
		s_Instance->ClearDepthImpl(depth);
	}

	void GfxDevice::WaitForFrame()
	{
		s_Instance->WaitForFrameImpl();
	}

	void GfxDevice::SwapBuffers()
	{
		s_Instance->SwapBuffersImpl();
	}

	void GfxDevice::SetViewport(int x, int y, int width, int height)
	{
		s_Instance->SetViewportImpl(x, y, width, height);
	}

	void GfxDevice::SetScissorRect(int x, int y, int width, int height)
	{
		s_Instance->SetScissorRectImpl(x, y, width, height);
	}

	void GfxDevice::ResizeBackbuffer(int width, int height)
	{
		s_Instance->ResizeBackbufferImpl(width, height);
	}

	uint32_t GfxDevice::GetViewCount()
	{
		return s_Instance->GetViewCountImpl();
	}

	void GfxDevice::SetViewCount(uint32_t count)
	{
		s_Instance->SetViewCountImpl(count);
	}

	void GfxDevice::SetDepthBias(uint32_t bias, float slopeBias)
	{
		s_Instance->SetDepthBiasImpl(bias, slopeBias);
	}

	bool GfxDevice::CreateVertexShader(const ByteData& vertexData, GfxVertexShader*& shader)
	{
		return s_Instance->CreateVertexShaderImpl(vertexData, shader);
	}

	bool GfxDevice::CreateGeometryShader(const ByteData& geometryData, GfxGeometryShader*& shader)
	{
		return s_Instance->CreateGeometryShaderImpl(geometryData, shader);
	}

	bool GfxDevice::CreateFragmentShader(const ByteData& fragmentData, GfxFragmentShader*& shader)
	{
		return s_Instance->CreateFragmentShaderImpl(fragmentData, shader);
	}

	bool GfxDevice::CreateComputeShader(const ByteData& computeData, GfxComputeShader*& shader)
	{
		return s_Instance->CreateComputeShaderImpl(computeData, shader);
	}

	bool GfxDevice::CreateRayTracingShader(const ByteData& rayTracingData, GfxRayTracingShader*& shader)
	{
		return s_Instance->CreateRayTracingShaderImpl(rayTracingData, shader);
	}

	bool GfxDevice::CreateBuffer(const BufferProperties& properties, GfxBuffer*& buffer)
	{
		return s_Instance->CreateBufferImpl(properties, buffer);
	}

	bool GfxDevice::CreateTexture(const TextureProperties& properties, GfxTexture*& texture)
	{
		return s_Instance->CreateTextureImpl(properties, texture);
	}

	bool GfxDevice::CreateBottomLevelAccelerationStructure(const BottomLevelAccelerationStructureProperties& properties, GfxBottomLevelAccelerationStructure*& accelerationStructure)
	{
		return s_Instance->CreateBottomLevelAccelerationStructureImpl(properties, accelerationStructure);
	}

	bool GfxDevice::CreateTopLevelAccelerationStructure(GfxTopLevelAccelerationStructure*& accelerationStructure)
	{
		return s_Instance->CreateTopLevelAccelerationStructureImpl(accelerationStructure);
	}

	void GfxDevice::Copy(GfxTexture* source, GfxTexture* target)
	{
		s_Instance->CopyImpl(source, target);
	}

	void GfxDevice::Copy(GfxTexture* source, GfxTexture* target, const Rectangle& area)
	{
		s_Instance->CopyImpl(source, target, area);
	}

	void GfxDevice::Copy(GfxTexture* source, GfxTexture* target, const Vector2Int& offset, const Rectangle& area)
	{
		s_Instance->CopyImpl(source, target, offset, area);
	}

	void GfxDevice::Copy(GfxTexture* source, GfxTexture* target, uint32_t sourceSlice, uint32_t targetSlice, uint32_t mipLevel)
	{
		s_Instance->CopyImpl(source, target, sourceSlice, targetSlice, mipLevel);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture)
	{
		s_Instance->SetRenderTargetImpl(&renderTexture, renderTexture == nullptr ? 0 : 1, nullptr, 0, 0);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture)
	{
		s_Instance->SetRenderTargetImpl(&renderTexture, renderTexture == nullptr ? 0 : 1, depthStencilTexture, 0, 0);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture, uint32_t arraySlice, uint32_t mipLevel)
	{
		s_Instance->SetRenderTargetImpl(&renderTexture, renderTexture == nullptr ? 0 : 1, nullptr, arraySlice, mipLevel);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, uint32_t arraySlice, uint32_t mipLevel)
	{
		s_Instance->SetRenderTargetImpl(&renderTexture, renderTexture == nullptr ? 0 : 1, depthStencilTexture, arraySlice, mipLevel);
	}

	void GfxDevice::SetRenderTarget(GfxTexture** renderTextures, uint32_t renderTexturesCount, GfxTexture* depthStencilTexture)
	{
		s_Instance->SetRenderTargetImpl(renderTextures, renderTexturesCount, depthStencilTexture, 0, 0);
	}

	void GfxDevice::SetGlobalBuffer(size_t id, GfxBuffer* buffer)
	{
		s_Instance->SetGlobalBufferImpl(id, buffer);
	}

	void GfxDevice::SetGlobalTexture(size_t id, GfxTexture* texture, uint32_t mip)
	{
		s_Instance->SetGlobalTextureImpl(id, texture, mip);
	}

	void GfxDevice::Draw(const GfxDrawingOperation& operation)
	{
		s_Instance->DrawImpl(operation);
	}

	void GfxDevice::Dispatch(ComputeShader* shader, uint32_t kernelIndex, uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ)
	{
		s_Instance->DispatchImpl(shader, kernelIndex, threadGroupsX, threadGroupsY, threadGroupsZ);
	}

	void GfxDevice::DispatchRays(RayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, uint32_t width, uint32_t height, uint32_t depth)
	{
		s_Instance->DispatchRaysImpl(shader, accelerationStructure, width, height, depth);
	}

	Matrix GfxDevice::GetGPUMatrix(const Matrix& viewProjection)
	{
		return s_Instance->GetGPUMatrixImpl(viewProjection);
	}

	GfxDevice* GfxDevice::GetInstance()
	{
		return s_Instance;
	}
}