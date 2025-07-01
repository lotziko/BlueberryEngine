#include "GfxDevice.h"

#include "GraphicsAPI.h"

#include "Blueberry\Graphics\Mesh.h"

#include "..\..\Concrete\DX11\GfxDeviceDX11.h"

namespace Blueberry
{
	bool GfxDevice::Initialize(int width, int height, void* data)
	{
		switch (GraphicsAPI::GetAPI())
		{
		case GraphicsAPI::API::None:
			BB_ERROR("API doesn't exist.");
			return false;
		case GraphicsAPI::API::DX11:
			s_Instance = new GfxDeviceDX11();
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

	void GfxDevice::ClearDepth(const float& depth)
	{
		s_Instance->ClearDepthImpl(depth);
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

	const uint32_t& GfxDevice::GetViewCount()
	{
		return s_Instance->GetViewCountImpl();
	}

	void GfxDevice::SetViewCount(const uint32_t& count)
	{
		s_Instance->SetViewCountImpl(count);
	}

	void GfxDevice::SetDepthBias(const uint32_t& bias, const float& slopeBias)
	{
		s_Instance->SetDepthBiasImpl(bias, slopeBias);
	}

	bool GfxDevice::CreateVertexShader(void* vertexData, GfxVertexShader*& shader)
	{
		return s_Instance->CreateVertexShaderImpl(vertexData, shader);
	}

	bool GfxDevice::CreateGeometryShader(void* geometryData, GfxGeometryShader*& shader)
	{
		return s_Instance->CreateGeometryShaderImpl(geometryData, shader);
	}

	bool GfxDevice::CreateFragmentShader(void* fragmentData, GfxFragmentShader*& shader)
	{
		return s_Instance->CreateFragmentShaderImpl(fragmentData, shader);
	}

	bool GfxDevice::CreateComputeShader(void* computeData, GfxComputeShader*& shader)
	{
		return s_Instance->CreateComputeShaderImpl(computeData, shader);
	}

	bool GfxDevice::CreateBuffer(const BufferProperties& properties, GfxBuffer*& buffer)
	{
		return s_Instance->CreateBufferImpl(properties, buffer);
	}

	bool GfxDevice::CreateTexture(const TextureProperties& properties, GfxTexture*& texture)
	{
		return s_Instance->CreateTextureImpl(properties, texture);
	}

	void GfxDevice::Copy(GfxTexture* source, GfxTexture* target)
	{
		s_Instance->CopyImpl(source, target);
	}

	void GfxDevice::Copy(GfxTexture* source, GfxTexture* target, const Rectangle& area)
	{
		s_Instance->CopyImpl(source, target, area);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture)
	{
		s_Instance->SetRenderTargetImpl(renderTexture, nullptr);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture)
	{
		s_Instance->SetRenderTargetImpl(renderTexture, depthStencilTexture);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture, const uint32_t& slice)
	{
		s_Instance->SetRenderTargetImpl(renderTexture, nullptr, slice);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, const uint32_t& slice)
	{
		s_Instance->SetRenderTargetImpl(renderTexture, depthStencilTexture, slice);
	}

	void GfxDevice::SetGlobalBuffer(const size_t& id, GfxBuffer* buffer)
	{
		s_Instance->SetGlobalBufferImpl(id, buffer);
	}

	void GfxDevice::SetGlobalTexture(const size_t& id, GfxTexture* texture)
	{
		s_Instance->SetGlobalTextureImpl(id, texture);
	}

	void GfxDevice::Draw(const GfxDrawingOperation& operation)
	{
		s_Instance->DrawImpl(operation);
	}

	void GfxDevice::Dispatch(GfxComputeShader* shader, const uint32_t& threadGroupsX, const uint32_t& threadGroupsY, const uint32_t& threadGroupsZ)
	{
		s_Instance->DispatchImpl(shader, threadGroupsX, threadGroupsY, threadGroupsZ);
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