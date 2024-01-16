#include "bbpch.h"
#include "GfxDeviceDX11.h"
#include "GfxShaderDX11.h"
#include "GfxBufferDX11.h"
#include "GfxTextureDX11.h"
#include "ImGuiRendererDX11.h"
#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Enums.h"

namespace Blueberry
{
	bool GfxDeviceDX11::InitializeImpl(int width, int height, void* data)
	{
		if (!InitializeDirectX(*(static_cast<HWND*>(data)), width, height))
			return false;

		return true;
	}

	void GfxDeviceDX11::ClearColorImpl(const Color& color) const
	{
		if (m_BindedRenderTarget == nullptr)
		{
			m_DeviceContext->ClearRenderTargetView(m_RenderTargetView.Get(), color);
		}
		else
		{
			m_DeviceContext->ClearRenderTargetView(m_BindedRenderTarget->m_RenderTargetView.Get(), color);
		}
	}

	void GfxDeviceDX11::ClearDepthImpl(const float& depth) const
	{
		if (m_BindedDepthStencil != nullptr)
		{
			m_DeviceContext->ClearDepthStencilView(m_BindedDepthStencil->m_DepthStencilView.Get(), D3D11_CLEAR_DEPTH, depth, 0);
		}
	}

	// TODO clear depth

	void GfxDeviceDX11::SwapBuffersImpl() const
	{
		m_SwapChain->Present(1, NULL);
	}

	void GfxDeviceDX11::SetViewportImpl(int x, int y, int width, int height)
	{
		D3D11_VIEWPORT viewport;
		ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

		viewport.TopLeftX = x;
		viewport.TopLeftY = y;
		viewport.Width = static_cast<FLOAT>(width);
		viewport.Height = static_cast<FLOAT>(height);
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;

		m_DeviceContext->RSSetViewports(1, &viewport);
	}

	void GfxDeviceDX11::ResizeBackbufferImpl(int width, int height)
	{
		m_DeviceContext->OMSetRenderTargets(0, 0, 0);
		m_RenderTargetView->Release();

		HRESULT hr;

		hr = m_SwapChain->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "ResizeBuffers failed."));
			return;
		}

		ID3D11Texture2D* backBuffer;
		hr = m_SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "GetBuffer failed."));
			return;
		}

		hr = m_Device->CreateRenderTargetView(backBuffer, NULL, m_RenderTargetView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
			return;
		}

		backBuffer->Release();

		m_DeviceContext->OMSetRenderTargets(1, m_RenderTargetView.GetAddressOf(), NULL);

		D3D11_VIEWPORT viewport;
		ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

		viewport.TopLeftX = 0;
		viewport.TopLeftY = 0;
		viewport.Width = static_cast<FLOAT>(width);
		viewport.Height = static_cast<FLOAT>(height);
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;

		m_DeviceContext->RSSetViewports(1, &viewport);
	}

	void GfxDeviceDX11::SetSurfaceTypeImpl(const SurfaceType& type)
	{
		m_DeviceContext->RSSetState(m_RasterizerState.Get());

		const float blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
		switch (type)
		{
		case SurfaceType::Opaque:
			m_DeviceContext->OMSetDepthStencilState(m_OpaqueDepthStencilState.Get(), 0);
			m_DeviceContext->OMSetBlendState(m_OpaqueBlendState.Get(), blendFactor, 0xffffffff);
			break;
		case SurfaceType::Transparent:
			m_DeviceContext->OMSetDepthStencilState(m_TransparentDepthStencilState.Get(), 0);
			m_DeviceContext->OMSetBlendState(m_TransparentBlendState.Get(), blendFactor, 0xffffffff);
			break;
		case SurfaceType::DepthTransparent:
			m_DeviceContext->OMSetDepthStencilState(m_OpaqueDepthStencilState.Get(), 0);
			m_DeviceContext->OMSetBlendState(m_TransparentBlendState.Get(), blendFactor, 0xffffffff);
			break;
		}
	}

	bool GfxDeviceDX11::CreateShaderImpl(void* vertexData, void* pixelData, GfxShader*& shader)
	{
		auto dxShader = new GfxShaderDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxShader->Initialize(vertexData, pixelData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateVertexBufferImpl(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer)
	{
		auto dxBuffer = new GfxVertexBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(layout, vertexCount))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateIndexBufferImpl(const UINT& indexCount, GfxIndexBuffer*& buffer)
	{
		auto dxBuffer = new GfxIndexBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(indexCount))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateConstantBufferImpl(const UINT& byteSize, GfxConstantBuffer*& buffer)
	{
		auto dxBuffer = new GfxConstantBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(byteSize))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) const
	{
		auto dxTexture = new GfxTextureDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxTexture->Create(properties))
		{
			return false;
		}
		texture = dxTexture;
		return true;
	}

	bool GfxDeviceDX11::CreateImGuiRendererImpl(ImGuiRenderer*& renderer) const
	{
		auto dxRenderer = new ImGuiRendererDX11(m_Hwnd, m_Device.Get(), m_DeviceContext.Get());
		renderer = dxRenderer;
		return true;
	}

	void GfxDeviceDX11::CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) const
	{
		D3D11_BOX src;
		src.left = area.x;
		src.top = area.y;
		src.right = area.x + area.width;
		src.bottom = area.y + area.height;
		src.front = 0;
		src.back = 1;

		m_DeviceContext->CopySubresourceRegion(static_cast<GfxTextureDX11*>(target)->m_Texture.Get(), 0, 0, 0, 0, static_cast<GfxTextureDX11*>(source)->m_Texture.Get(), 0, &src);
	}

	void GfxDeviceDX11::SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture)
	{
		if (renderTexture == nullptr)
		{
			m_DeviceContext->OMSetRenderTargets(1, m_RenderTargetView.GetAddressOf(), NULL);
			m_BindedRenderTarget = nullptr;
			m_BindedDepthStencil = nullptr;
		}
		else
		{
			auto dxRenderTarget = static_cast<GfxTextureDX11*>(renderTexture);
			m_BindedRenderTarget = dxRenderTarget;
			if (depthStencilTexture == nullptr)
			{
				m_DeviceContext->OMSetRenderTargets(1, dxRenderTarget->m_RenderTargetView.GetAddressOf(), NULL);
				m_BindedDepthStencil = nullptr;
			}
			else
			{
				auto dxDepthStencil = static_cast<GfxTextureDX11*>(depthStencilTexture);
				m_DeviceContext->OMSetRenderTargets(1, dxRenderTarget->m_RenderTargetView.GetAddressOf(), dxDepthStencil->m_DepthStencilView.Get());
				m_BindedDepthStencil = dxDepthStencil;
			}
		}
	}

	void GfxDeviceDX11::SetGlobalConstantBufferImpl(const std::size_t& id, GfxConstantBuffer* buffer)
	{
		auto dxConstantBuffer = static_cast<GfxConstantBufferDX11*>(buffer);
		m_BindedConstantBuffers.insert_or_assign(id, dxConstantBuffer);
	}

	void GfxDeviceDX11::SetGlobalTextureImpl(const std::size_t& id, GfxTexture* texture)
	{
		auto dxTexture = static_cast<GfxTextureDX11*>(texture);
		m_BindedTextures.insert_or_assign(id, dxTexture);
	}

	D3D11_PRIMITIVE_TOPOLOGY GetPrimitiveTopology(const Topology& topology)
	{
		switch (topology)
		{
		case Topology::Unknown:			return D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
		case Topology::LineList:		return D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
		case Topology::LineStrip:		return D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP;
		case Topology::TriangleList:	return D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		}
	}

	void GfxDeviceDX11::DrawImpl(const GfxDrawingOperation& operation) const
	{
		if (!operation.IsValid())
		{
			return;
		}

		// TODO check if shader variant/material/mesh is the same to skip some bindings
		auto dxShader = static_cast<GfxShaderDX11*>(operation.shader);
		m_DeviceContext->IASetInputLayout(dxShader->m_InputLayout.Get());
		m_DeviceContext->VSSetShader(dxShader->m_VertexShader.Get(), NULL, 0);
		m_DeviceContext->PSSetShader(dxShader->m_PixelShader.Get(), NULL, 0);

		m_DeviceContext->IASetPrimitiveTopology(GetPrimitiveTopology(operation.topology));

		auto textureVector = operation.textures;
		for (int i = 0; i < textureVector->size(); i++)
		{
			auto pair = textureVector->at(i);
			auto dxTexture = static_cast<GfxTextureDX11*>(pair.second);
			auto slot = dxShader->m_TextureSlots.find(pair.first);
			if (slot != dxShader->m_TextureSlots.end())
			{
				UINT slotIndex = slot->second;
				m_DeviceContext->PSSetShaderResources(slotIndex, 1, dxTexture->m_ResourceView.GetAddressOf());
				m_DeviceContext->PSSetSamplers(slotIndex, 1, dxTexture->m_SamplerState.GetAddressOf());
			}
		}

		std::map<std::size_t, UINT>::iterator it;

		auto textureMap = dxShader->m_TextureSlots;
		for (it = textureMap.begin(); it != textureMap.end(); it++)
		{
			auto pair = m_BindedTextures.find(it->first);
			if (pair != m_BindedTextures.end())
			{
				UINT slotIndex = it->second;
				auto dxTexture = pair->second;
				m_DeviceContext->PSSetShaderResources(slotIndex, 1, dxTexture->m_ResourceView.GetAddressOf());
				m_DeviceContext->PSSetSamplers(slotIndex, 1, dxTexture->m_SamplerState.GetAddressOf());
			}
		}

		auto dxVertexBuffer = static_cast<GfxVertexBufferDX11*>(operation.vertexBuffer);
		m_DeviceContext->IASetVertexBuffers(0, 1, dxVertexBuffer->m_Buffer.GetAddressOf(), &dxVertexBuffer->m_Stride, &dxVertexBuffer->m_Offset);
	
		auto dxIndexBuffer = static_cast<GfxIndexBufferDX11*>(operation.indexBuffer);
		m_DeviceContext->IASetIndexBuffer(dxIndexBuffer->m_Buffer.Get(), DXGI_FORMAT_R32_UINT, 0);

		auto bufferMap = dxShader->m_VertexConstantBufferSlots;
		int offset = 0;
		for (it = bufferMap.begin(); it != bufferMap.end(); it++)
		{
			auto pair = m_BindedConstantBuffers.find(it->first);
			if (pair != m_BindedConstantBuffers.end())
			{
				m_DeviceContext->VSSetConstantBuffers(offset, 1, pair->second->m_Buffer.GetAddressOf());
			}
			++offset;
		}

		bufferMap = dxShader->m_PixelConstantBufferSlots;
		offset = 0;
		for (it = bufferMap.begin(); it != bufferMap.end(); it++)
		{
			auto pair = m_BindedConstantBuffers.find(it->first);
			if (pair != m_BindedConstantBuffers.end())
			{
				m_DeviceContext->PSSetConstantBuffers(offset, 1, pair->second->m_Buffer.GetAddressOf());
			}
			++offset;
		}

		m_DeviceContext->DrawIndexed(operation.indexCount, operation.indexOffset, 0);
	}

	Matrix GfxDeviceDX11::GetGPUMatrixImpl(const Matrix& viewProjection) const
	{
		Matrix copy;
		viewProjection.Transpose(copy);
		return copy;
	}

	bool GfxDeviceDX11::InitializeDirectX(HWND hwnd, int width, int height)
	{
		m_Hwnd = hwnd;

		DXGI_SWAP_CHAIN_DESC scd;
		ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));

		scd.BufferDesc.Width = width;
		scd.BufferDesc.Height = height;
		scd.BufferDesc.RefreshRate.Numerator = 60;
		scd.BufferDesc.RefreshRate.Denominator = 1;
		scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		scd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
		scd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

		scd.SampleDesc.Count = 1;
		scd.SampleDesc.Quality = 0;

		scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		scd.BufferCount = 1;
		scd.OutputWindow = hwnd;
		scd.Windowed = TRUE;
		scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
		scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

		HRESULT hr;
		hr = D3D11CreateDeviceAndSwapChain(
			NULL,
			D3D_DRIVER_TYPE_HARDWARE, //hardware driver
			NULL, //software driver
			D3D11_CREATE_DEVICE_DEBUG, //no flags
			NULL, //feature levels
			0, //no feature levels
			D3D11_SDK_VERSION,
			&scd, //swapchain description
			m_SwapChain.GetAddressOf(), //m_SwapChain address
			m_Device.GetAddressOf(), //m_Device address
			NULL, //supported feature level
			m_DeviceContext.GetAddressOf() //m_DeviceContext address
		);

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating swapchain."));
			return false;
		}

		ID3D11Texture2D* backBuffer;
		hr = m_SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "GetBuffer failed."));
			return false;
		}

		hr = m_Device->CreateRenderTargetView(backBuffer, NULL, m_RenderTargetView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
			return false;
		}

		backBuffer->Release();

		m_DeviceContext->OMSetRenderTargets(1, m_RenderTargetView.GetAddressOf(), NULL);

		D3D11_VIEWPORT viewport;
		ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

		viewport.TopLeftX = 0;
		viewport.TopLeftY = 0;
		viewport.Width = static_cast<FLOAT>(width);
		viewport.Height = static_cast<FLOAT>(height);
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;

		m_DeviceContext->RSSetViewports(1, &viewport);

		D3D11_RASTERIZER_DESC rasterizerDesc;
		ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

		rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
		rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE;

		hr = m_Device->CreateRasterizerState(&rasterizerDesc, m_RasterizerState.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create rasterizer state."));
			return false;
		}

		// Opaque
		{
			D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
			ZeroMemory(&depthStencilDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));

			depthStencilDesc.DepthEnable = true;
			depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
			depthStencilDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;

			hr = m_Device->CreateDepthStencilState(&depthStencilDesc, m_OpaqueDepthStencilState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil state."));
				return false;
			}

			D3D11_BLEND_DESC blendDesc;
			ZeroMemory(&blendDesc, sizeof(D3D11_BLEND_DESC));

			blendDesc.AlphaToCoverageEnable = false;
			blendDesc.RenderTarget[0].BlendEnable = false;
			blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
			blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
			blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
			blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
			blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
			blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
			blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

			hr = m_Device->CreateBlendState(&blendDesc, m_OpaqueBlendState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create blend state."));
				return false;
			}
		}

		// Transparent
		{
			D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
			ZeroMemory(&depthStencilDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));

			depthStencilDesc.DepthEnable = false;
			depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
			depthStencilDesc.DepthFunc = D3D11_COMPARISON_LESS;

			hr = m_Device->CreateDepthStencilState(&depthStencilDesc, m_TransparentDepthStencilState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil state."));
				return false;
			}

			D3D11_BLEND_DESC blendDesc;
			ZeroMemory(&blendDesc, sizeof(D3D11_BLEND_DESC));

			blendDesc.AlphaToCoverageEnable = false;
			blendDesc.RenderTarget[0].BlendEnable = true;
			blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
			blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
			blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
			blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
			blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
			blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
			blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

			hr = m_Device->CreateBlendState(&blendDesc, m_TransparentBlendState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create blend state."));
				return false;
			}
		}

		m_BindedRenderTarget = nullptr;
		m_BindedDepthStencil = nullptr;

		BB_INFO("DirectX initialized successful.");

		return true;
	}
}