#include "bbpch.h"
#include "GfxDevice.h"

#include "GraphicsAPI.h"

#include "Blueberry\Graphics\Mesh.h"

#include "Concrete\DX11\GfxDeviceDX11.h"

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

	void GfxDevice::ResizeBackbuffer(int width, int height)
	{
		s_Instance->ResizeBackbufferImpl(width, height);
	}

	void GfxDevice::SetCullMode(const CullMode& mode)
	{
		s_Instance->SetCullModeImpl(mode);
	}

	void GfxDevice::SetSurfaceType(const SurfaceType& type)
	{
		s_Instance->SetSurfaceTypeImpl(type);
	}

	bool GfxDevice::CreateShader(void* vertexData, void* pixelData, GfxShader*& shader)
	{
		return s_Instance->CreateShaderImpl(vertexData, pixelData, shader);
	}

	bool GfxDevice::CreateComputeShader(void* computeData, GfxComputeShader*& shader)
	{
		return s_Instance->CreateComputeShaderImpl(computeData, shader);
	}

	bool GfxDevice::CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer)
	{
		return s_Instance->CreateVertexBufferImpl(layout, vertexCount, buffer);
	}

	bool GfxDevice::CreateIndexBuffer(const UINT& indexCount, GfxIndexBuffer*& buffer)
	{
		return s_Instance->CreateIndexBufferImpl(indexCount, buffer);
	}

	bool GfxDevice::CreateConstantBuffer(const UINT& byteCount, GfxConstantBuffer*& buffer)
	{
		return s_Instance->CreateConstantBufferImpl(byteCount, buffer);
	}

	bool GfxDevice::CreateComputeBuffer(const UINT& elementCount, const UINT& elementSize, GfxComputeBuffer*& buffer)
	{
		return s_Instance->CreateComputeBufferImpl(elementCount, elementSize, buffer);
	}

	bool GfxDevice::CreateTexture(const TextureProperties& properties, GfxTexture*& texture)
	{
		return s_Instance->CreateTextureImpl(properties, texture);
	}

	bool GfxDevice::CreateImGuiRenderer(ImGuiRenderer*& renderer)
	{
		return s_Instance->CreateImGuiRendererImpl(renderer);
	}

	void GfxDevice::Copy(GfxTexture* source, GfxTexture* target, const Rectangle& area)
	{
		s_Instance->CopyImpl(source, target, area);
	}

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture, GfxTexture* depthStencilTexture)
	{
		s_Instance->SetRenderTargetImpl(renderTexture, depthStencilTexture);
	}

	void GfxDevice::SetGlobalConstantBuffer(const std::size_t& id, GfxConstantBuffer* buffer)
	{
		s_Instance->SetGlobalConstantBufferImpl(id, buffer);
	}

	void GfxDevice::SetGlobalTexture(const std::size_t& id, GfxTexture* texture)
	{
		s_Instance->SetGlobalTextureImpl(id, texture);
	}

	void GfxDevice::Draw(const GfxDrawingOperation& operation)
	{
		s_Instance->DrawImpl(operation);
	}

	void GfxDevice::Dispatch(GfxComputeShader*& shader, const UINT& threadGroupsX, const UINT& threadGroupsY, const UINT& threadGroupsZ)
	{
		s_Instance->DispatchImpl(shader, threadGroupsX, threadGroupsY, threadGroupsZ);
	}

	Matrix GfxDevice::GetGPUMatrix(const Matrix& viewProjection)
	{
		return s_Instance->GetGPUMatrixImpl(viewProjection);
	}
}