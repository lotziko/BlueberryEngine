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
	GfxDeviceDX11::GfxDeviceDX11()
	{
	}

	bool GfxDeviceDX11::Initialize(int width, int height, void* data)
	{
		if (!InitializeDirectX(*(static_cast<HWND*>(data)), width, height))
			return false;

		return true;
	}

	void GfxDeviceDX11::ClearColor(const Color& color) const
	{
		if (m_BindedRenderTarget == nullptr)
		{
			m_DeviceContext->ClearRenderTargetView(m_RenderTargetView.Get(), color);
		}
		else
		{
			m_BindedRenderTarget->Clear(color);
		}
	}

	void GfxDeviceDX11::SwapBuffers() const
	{
		m_SwapChain->Present(1, NULL);
	}

	void GfxDeviceDX11::SetViewport(int x, int y, int width, int height)
	{
		D3D11_VIEWPORT viewport;
		ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

		viewport.TopLeftX = x;
		viewport.TopLeftY = y;
		viewport.Width = static_cast<FLOAT>(width);
		viewport.Height = static_cast<FLOAT>(height);

		m_DeviceContext->RSSetViewports(1, &viewport);
	}

	void GfxDeviceDX11::ResizeBackbuffer(int width, int height)
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

		m_DeviceContext->RSSetViewports(1, &viewport);
	}

	bool GfxDeviceDX11::CreateShader(const std::wstring& shaderPath, Ref<GfxShader>& shader)
	{
		auto dxShader = CreateRef<GfxShaderDX11>(m_Device.Get(), m_DeviceContext.Get());
		if (!dxShader->Compile(shaderPath))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateShader(const std::wstring& vertexShaderPath, const std::wstring& pixelShaderPath, Ref<GfxShader>& shader)
	{
		auto dxShader = CreateRef<GfxShaderDX11>(m_Device.Get(), m_DeviceContext.Get());
		if (!dxShader->Initialize(vertexShaderPath, pixelShaderPath))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, Ref<GfxVertexBuffer>& buffer)
	{
		auto dxBuffer = CreateRef<GfxVertexBufferDX11>(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(layout, vertexCount))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateIndexBuffer(const UINT& indexCount, Ref<GfxIndexBuffer>& buffer)
	{
		auto dxBuffer = CreateRef<GfxIndexBufferDX11>(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(indexCount))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateConstantBuffer(const UINT& byteSize, Ref<GfxConstantBuffer>& buffer)
	{
		auto dxBuffer = CreateRef<GfxConstantBufferDX11>(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(byteSize))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateTexture(const TextureProperties& properties, Ref<GfxTexture>& texture) const
	{
		auto dxTexture = CreateRef<GfxTextureDX11>(m_Device.Get(), m_DeviceContext.Get());
		if (!dxTexture->Create(properties))
		{
			return false;
		}
		texture = dxTexture;
		return true;
	}

	bool GfxDeviceDX11::CreateImGuiRenderer(Ref<ImGuiRenderer>& renderer) const
	{
		auto dxRenderer = CreateRef<ImGuiRendererDX11>(m_Hwnd, m_Device.Get(), m_DeviceContext.Get());
		renderer = dxRenderer;
		return true;
	}

	void GfxDeviceDX11::SetRenderTarget(GfxTexture* renderTexture)
	{
		if (renderTexture == nullptr)
		{
			m_DeviceContext->OMSetRenderTargets(1, m_RenderTargetView.GetAddressOf(), NULL);
			m_BindedRenderTarget = nullptr;
		}
		else
		{
			auto dxRenderTarget = static_cast<GfxTextureDX11*>(renderTexture);
			m_DeviceContext->OMSetRenderTargets(1, dxRenderTarget->m_RenderTargetView.GetAddressOf(), NULL);
			m_BindedRenderTarget = dxRenderTarget;
		}
	}

	void GfxDeviceDX11::SetGlobalConstantBuffer(const std::size_t& id, GfxConstantBuffer* buffer)
	{
		if (m_BindedConstantBuffers.count(id) == 0)
		{
			auto dxConstantBuffer = static_cast<GfxConstantBufferDX11*>(buffer);
			m_BindedConstantBuffers.insert({ id, dxConstantBuffer });
		}
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

	void GfxDeviceDX11::Draw(const GfxDrawingOperation& operation) const
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

		auto dxVertexBuffer = static_cast<GfxVertexBufferDX11*>(operation.vertexBuffer);
		m_DeviceContext->IASetVertexBuffers(0, 1, dxVertexBuffer->m_Buffer.GetAddressOf(), &dxVertexBuffer->m_Stride, &dxVertexBuffer->m_Offset);
	
		auto dxIndexBuffer = static_cast<GfxIndexBufferDX11*>(operation.indexBuffer);
		m_DeviceContext->IASetIndexBuffer(dxIndexBuffer->m_Buffer.Get(), DXGI_FORMAT_R32_UINT, 0);

		auto bufferMap = dxShader->m_ConstantBufferSlots;
		int offset = 0;
		std::map<std::size_t, UINT>::iterator it;
		for (it = bufferMap.begin(); it != bufferMap.end(); it++)
		{
			auto pair = m_BindedConstantBuffers.find(it->first);
			if (pair != m_BindedConstantBuffers.end())
			{
				m_DeviceContext->VSSetConstantBuffers(offset, 1, pair->second->m_Buffer.GetAddressOf());
			}
			++offset;
		}

		m_DeviceContext->RSSetState(m_RasterizerState.Get());
		m_DeviceContext->OMSetDepthStencilState(m_DepthStencilState.Get(), 0);
		m_DeviceContext->DrawIndexed(operation.indexCount, 0, 0);
	}

	Matrix GfxDeviceDX11::GetGPUMatrix(const Matrix& viewProjection) const
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

		D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
		ZeroMemory(&depthStencilDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));

		depthStencilDesc.DepthEnable = FALSE;
		depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;

		hr = m_Device->CreateDepthStencilState(&depthStencilDesc, m_DepthStencilState.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil state."));
			return false;
		}

		m_BindedRenderTarget = nullptr;

		BB_INFO("DirectX initialized successful.");

		return true;
	}
}