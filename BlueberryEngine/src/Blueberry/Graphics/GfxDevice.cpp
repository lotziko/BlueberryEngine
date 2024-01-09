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

	void GfxDevice::ClearColor(const Color& color)
	{
		s_Instance->ClearColorImpl(color);
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

	bool GfxDevice::CreateShader(void* vertexData, void* pixelData, GfxShader*& shader)
	{
		return s_Instance->CreateShaderImpl(vertexData, pixelData, shader);
	}

	bool GfxDevice::CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer)
	{
		return s_Instance->CreateVertexBufferImpl(layout, vertexCount, buffer);
	}

	bool GfxDevice::CreateIndexBuffer(const UINT& indexCount, GfxIndexBuffer*& buffer)
	{
		return s_Instance->CreateIndexBufferImpl(indexCount, buffer);
	}

	bool GfxDevice::CreateConstantBuffer(const UINT& byteSize, GfxConstantBuffer*& buffer)
	{
		return s_Instance->CreateConstantBufferImpl(byteSize, buffer);
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

	void GfxDevice::SetRenderTarget(GfxTexture* renderTexture)
	{
		s_Instance->SetRenderTargetImpl(renderTexture);
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

	Matrix GfxDevice::GetGPUMatrix(const Matrix& viewProjection)
	{
		return s_Instance->GetGPUMatrixImpl(viewProjection);
	}
}