#include "bbpch.h"
#include "GfxDeviceDX11.h"
#include "GfxShaderDX11.h"
#include "GfxComputeShaderDX11.h"
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

	void GfxDeviceDX11::SwapBuffersImpl() const
	{
		m_SwapChain->Present(1, NULL);

		// Clear
		ID3D11ShaderResourceView* emptySRV[1] = { nullptr };
		ID3D11SamplerState* emptySampler[1] = { nullptr };
		m_DeviceContext->PSSetShaderResources(0, 1, emptySRV);
		m_DeviceContext->PSSetSamplers(0, 1, emptySampler);
	}

	void GfxDeviceDX11::SetViewportImpl(int x, int y, int width, int height)
	{
		D3D11_VIEWPORT viewport;
		ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

		viewport.TopLeftX = static_cast<FLOAT>(x);
		viewport.TopLeftY = static_cast<FLOAT>(y);
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

	bool GfxDeviceDX11::CreateVertexShaderImpl(void* vertexData, GfxVertexShader*& shader)
	{
		auto dxShader = new GfxVertexShaderDX11();
		if (!dxShader->Initialize(m_Device.Get(), vertexData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateGeometryShaderImpl(void* geometryData, GfxGeometryShader*& shader)
	{
		auto dxShader = new GfxGeometryShaderDX11();
		if (!dxShader->Initialize(m_Device.Get(), geometryData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateFragmentShaderImpl(void* fragmentData, GfxFragmentShader*& shader)
	{
		auto dxShader = new GfxFragmentShaderDX11();
		if (!dxShader->Initialize(m_Device.Get(), fragmentData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateComputeShaderImpl(void* computeData, GfxComputeShader*& shader)
	{
		auto dxShader = new GfxComputeShaderDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxShader->Initialize(computeData))
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

	bool GfxDeviceDX11::CreateConstantBufferImpl(const UINT& byteCount, GfxConstantBuffer*& buffer)
	{
		auto dxBuffer = new GfxConstantBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(byteCount))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateComputeBufferImpl(const UINT& elementCount, const UINT& elementSize, GfxComputeBuffer*& buffer)
	{
		auto dxBuffer = new GfxComputeBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(elementCount, elementSize))
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

	void GfxDeviceDX11::ReadImpl(GfxTexture* source, void* target, const Rectangle& area) const
	{
		auto dxTexture = static_cast<GfxTextureDX11*>(source);
		m_DeviceContext->CopySubresourceRegion(dxTexture->m_StagingTexture.Get(), 0, 0, 0, 0, dxTexture->m_Texture.Get(), 0, NULL);

		D3D11_MAPPED_SUBRESOURCE mappedTexture;
		ZeroMemory(&mappedTexture, sizeof(D3D11_MAPPED_SUBRESOURCE));
		HRESULT hr = m_DeviceContext->Map(dxTexture->m_StagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedTexture);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get texture data."));
			return;
		}
		// TODO handle texture formats
		for (int i = 0; i < area.height; i++)
		{
			size_t pixelSize = sizeof(char) * 4;
			size_t offset = ((area.y + i) * dxTexture->m_Width + area.x) * pixelSize;
			char* copyPtr = static_cast<char*>(mappedTexture.pData) + offset;
			char* targetPtr = static_cast<char*>(target) + (area.width * pixelSize * i);
			memcpy(targetPtr, copyPtr, area.width * pixelSize);
		}
		m_DeviceContext->Unmap(dxTexture->m_StagingTexture.Get(), 0);
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
		default:						return D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
		}
	}

	void GfxDeviceDX11::DrawImpl(const GfxDrawingOperation& operation)
	{
		if (!operation.IsValid())
		{
			return;
		}

		GfxRenderState* renderState = operation.renderState;
		SetCullMode(renderState->cullMode);
		SetBlendMode(renderState->blendSrc, renderState->blendDst);
		SetZWrite(renderState->zWrite);
		SetTopology(operation.topology);

		// TODO check if shader variant/material/mesh is the same to skip some bindings

		if (operation.materialId != m_MaterialId)
		{
			m_MaterialId = operation.materialId;

			auto dxVertexShader = static_cast<GfxVertexShaderDX11*>(renderState->vertexShader);
			if (dxVertexShader != m_VertexShader)
			{
				m_VertexShader = dxVertexShader;

				m_DeviceContext->IASetInputLayout(dxVertexShader->m_InputLayout.Get());
				m_DeviceContext->VSSetShader(dxVertexShader->m_Shader.Get(), NULL, 0);

				std::fill_n(m_ConstantBuffers, 8, nullptr);

				// Bind constant buffers
				auto bufferMap = dxVertexShader->m_ConstantBufferSlots;
				for (auto it = bufferMap.begin(); it != bufferMap.end(); it++)
				{
					auto pair = m_BindedConstantBuffers.find(it->first);
					if (pair != m_BindedConstantBuffers.end())
					{
						m_ConstantBuffers[it->second] = pair->second->m_Buffer.Get();
					}
				}

				m_DeviceContext->VSSetConstantBuffers(0, 8, m_ConstantBuffers);
			}

			auto dxGeometryShader = static_cast<GfxGeometryShaderDX11*>(renderState->geometryShader);
			if (dxGeometryShader != m_GeometryShader)
			{
				m_GeometryShader = dxGeometryShader;

				m_DeviceContext->GSSetShader(dxGeometryShader == nullptr ? NULL : dxGeometryShader->m_Shader.Get(), NULL, 0);
			}

			auto dxFragmentShader = static_cast<GfxFragmentShaderDX11*>(renderState->fragmentShader);
			if (dxFragmentShader != m_FragmentShader)
			{
				m_DeviceContext->PSSetShader(dxFragmentShader->m_Shader.Get(), NULL, 0);

				std::fill_n(m_ConstantBuffers, 8, nullptr);

				// Bind constant buffers
				auto bufferMap = dxFragmentShader->m_ConstantBufferSlots;
				for (auto it = bufferMap.begin(); it != bufferMap.end(); it++)
				{
					auto pair = m_BindedConstantBuffers.find(it->first);
					if (pair != m_BindedConstantBuffers.end())
					{
						m_ConstantBuffers[it->second] = pair->second->m_Buffer.Get();
					}
				}

				m_DeviceContext->PSSetConstantBuffers(0, 8, m_ConstantBuffers);
				
				std::fill_n(m_ShaderResourceViews, 16, nullptr);
				std::fill_n(m_Samplers, 16, nullptr);

				// Bind material textures
				auto textureMap = dxFragmentShader->m_TextureSlots;
				for (int i = 0; i < renderState->fragmentTextureCount; i++)
				{
					GfxRenderState::TextureInfo info = renderState->fragmentTextures[i];
					auto dxTexture = static_cast<GfxTextureDX11*>(info.texture);
					m_ShaderResourceViews[info.textureSlot] = dxTexture->m_ResourceView.Get();
					if (info.samplerSlot != -1)
					{
						m_Samplers[info.samplerSlot] = dxTexture->m_SamplerState.Get();
					}
				}

				// Bind global textures
				for (auto it = textureMap.begin(); it != textureMap.end(); it++)
				{
					auto pair = m_BindedTextures.find(it->first);
					if (pair != m_BindedTextures.end())
					{
						UINT textureSlotIndex = it->second.first;
						UINT samplerSlotIndex = it->second.second;
						auto dxTexture = pair->second;
						m_ShaderResourceViews[textureSlotIndex] = dxTexture->m_ResourceView.Get();
						if (samplerSlotIndex != -1)
						{
							m_Samplers[samplerSlotIndex] = dxTexture->m_SamplerState.Get();
						}
					}
				}

				m_DeviceContext->PSSetShaderResources(0, 16, m_ShaderResourceViews);
				m_DeviceContext->PSSetSamplers(0, 16, m_Samplers);
			}
		}

		// TODO Check mesh too
		auto dxVertexBuffer = static_cast<GfxVertexBufferDX11*>(operation.vertexBuffer);
		m_DeviceContext->IASetVertexBuffers(0, 1, dxVertexBuffer->m_Buffer.GetAddressOf(), &dxVertexBuffer->m_Stride, &dxVertexBuffer->m_Offset);

		auto dxIndexBuffer = static_cast<GfxIndexBufferDX11*>(operation.indexBuffer);
		m_DeviceContext->IASetIndexBuffer(dxIndexBuffer->m_Buffer.Get(), DXGI_FORMAT_R32_UINT, 0);

		m_DeviceContext->DrawIndexed(operation.indexCount, operation.indexOffset, 0);
	}

	void GfxDeviceDX11::DispatchImpl(GfxComputeShader*& shader, const UINT& threadGroupsX, const UINT& threadGroupsY, const UINT& threadGroupsZ) const
	{
		auto dxShader = static_cast<GfxComputeShaderDX11*>(shader);

		m_DeviceContext->CSSetShader(dxShader->m_ComputeShader.Get(), NULL, 0);
		m_DeviceContext->Dispatch(threadGroupsX, threadGroupsY, threadGroupsZ);
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

		// None
		{
			D3D11_RASTERIZER_DESC rasterizerDesc;
			ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

			rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
			rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE;
			rasterizerDesc.MultisampleEnable = true;
			rasterizerDesc.AntialiasedLineEnable = true;

			hr = m_Device->CreateRasterizerState(&rasterizerDesc, m_CullNoneRasterizerState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create cull none rasterizer state."));
				return false;
			}
		}

		// Front
		{
			D3D11_RASTERIZER_DESC rasterizerDesc;
			ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

			rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
			rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_FRONT;
			rasterizerDesc.MultisampleEnable = true;
			rasterizerDesc.AntialiasedLineEnable = true;

			hr = m_Device->CreateRasterizerState(&rasterizerDesc, m_CullFrontRasterizerState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create cull front rasterizer state."));
				return false;
			}
		}

		// Back
		{
			D3D11_RASTERIZER_DESC rasterizerDesc;
			ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

			rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
			rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_BACK;
			rasterizerDesc.MultisampleEnable = true;
			rasterizerDesc.AntialiasedLineEnable = true;

			hr = m_Device->CreateRasterizerState(&rasterizerDesc, m_CullBackRasterizerState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create cull back rasterizer state."));
				return false;
			}
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

		SetCullMode(CullMode::None);
		SetBlendMode(BlendMode::One, BlendMode::Zero);
		SetZWrite(ZWrite::On);

		m_BindedRenderTarget = nullptr;
		m_BindedDepthStencil = nullptr;

		BB_INFO("DirectX initialized successful.");

		return true;
	}

	void GfxDeviceDX11::SetCullMode(const CullMode& mode)
	{
		if (mode == m_CullMode)
		{
			return;
		}
		m_CullMode = mode;

		switch (mode)
		{
		case CullMode::None:
			m_DeviceContext->RSSetState(m_CullNoneRasterizerState.Get());
			break;
		case CullMode::Front:
			m_DeviceContext->RSSetState(m_CullFrontRasterizerState.Get());
			break;
		case CullMode::Back:
			m_DeviceContext->RSSetState(m_CullBackRasterizerState.Get());
			break;
		}
	}

	void GfxDeviceDX11::SetBlendMode(const BlendMode& blendSrc, const BlendMode& blendDst)
	{
		if (blendSrc == m_BlendSrc && blendDst == m_BlendDst)
		{
			return;
		}
		m_BlendSrc = blendSrc;
		m_BlendDst = blendDst;

		const float blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
		if (blendSrc == BlendMode::One && blendDst == BlendMode::Zero)
		{
			m_DeviceContext->OMSetBlendState(m_OpaqueBlendState.Get(), blendFactor, 0xffffffff);
		}
		else if (blendSrc == BlendMode::SrcAlpha && blendDst == BlendMode::OneMinusSrcAlpha)
		{
			m_DeviceContext->OMSetBlendState(m_TransparentBlendState.Get(), blendFactor, 0xffffffff);
		}
	}

	void GfxDeviceDX11::SetZWrite(const ZWrite& zWrite)
	{
		if (zWrite == m_ZWrite)
		{
			return;
		}
		m_ZWrite = zWrite;

		if (zWrite == ZWrite::On)
		{
			m_DeviceContext->OMSetDepthStencilState(m_OpaqueDepthStencilState.Get(), 0);
		}
		else
		{
			m_DeviceContext->OMSetDepthStencilState(m_TransparentDepthStencilState.Get(), 0);
		}
	}

	void GfxDeviceDX11::SetTopology(const Topology& topology)
	{
		if (topology == m_Topology)
		{
			return;
		}
		m_Topology = topology;

		m_DeviceContext->IASetPrimitiveTopology(GetPrimitiveTopology(topology));
	}
}