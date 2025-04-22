#include "bbpch.h"
#include "GfxDeviceDX11.h"
#include "GfxShaderDX11.h"
#include "GfxComputeShaderDX11.h"
#include "GfxBufferDX11.h"
#include "GfxTextureDX11.h"
#include "ImGuiRendererDX11.h"
#include "HBAORendererDX11.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Tools\CRCHelper.h"

namespace Blueberry
{
	bool GfxDeviceDX11::InitializeImpl(int width, int height, void* data)
	{
		if (!InitializeDirectX(*(static_cast<HWND*>(data)), width, height))
			return false;

		m_StateCache = GfxRenderStateCacheDX11(this);
		m_LayoutCache = GfxInputLayoutCacheDX11();
		m_BindedTextures.reserve(64);

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

	void GfxDeviceDX11::SwapBuffersImpl()
	{
		m_SwapChain->Present(1, NULL);
		Clear();
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

	void GfxDeviceDX11::SetScissorRectImpl(int x, int y, int width, int height)
	{
		if (width > 0)
		{
			D3D11_RECT rect;

			rect.left = x;
			rect.right = x + width;
			rect.top = y;
			rect.bottom = y + height;

			m_DeviceContext->RSSetScissorRects(1, &rect);
		}
		else
		{
			m_DeviceContext->RSSetScissorRects(0, NULL);
		}
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

	const uint32_t& GfxDeviceDX11::GetViewCountImpl()
	{
		return m_ViewCount;
	}

	void GfxDeviceDX11::SetViewCountImpl(const uint32_t& count)
	{
		m_ViewCount = count;
	}

	void GfxDeviceDX11::SetDepthBiasImpl(const uint32_t& bias, const float& slopeBias)
	{
		m_DepthBias = bias;
		m_SlopeDepthBias = slopeBias;
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

	bool GfxDeviceDX11::CreateVertexBufferImpl(const uint32_t& vertexCount, const uint32_t& vertexSize, GfxVertexBuffer*& buffer)
	{
		auto dxBuffer = new GfxVertexBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(vertexCount, vertexSize))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateIndexBufferImpl(const uint32_t& indexCount, GfxIndexBuffer*& buffer)
	{
		auto dxBuffer = new GfxIndexBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(indexCount))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateConstantBufferImpl(const uint32_t& byteCount, GfxConstantBuffer*& buffer)
	{
		auto dxBuffer = new GfxConstantBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(byteCount))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateStructuredBufferImpl(const uint32_t& elementCount, const uint32_t& elementSize, GfxStructuredBuffer*& buffer)
	{
		auto dxBuffer = new GfxStructuredBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(elementCount, elementSize))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateComputeBufferImpl(const uint32_t& elementCount, const uint32_t& elementSize, GfxComputeBuffer*& buffer)
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

	void GfxDeviceDX11::CopyImpl(GfxTexture* source, GfxTexture* target) const
	{
		m_DeviceContext->CopySubresourceRegion(static_cast<GfxTextureDX11*>(target)->m_Texture.Get(), 0, 0, 0, 0, static_cast<GfxTextureDX11*>(source)->m_Texture.Get(), 0, NULL);
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

	void GfxDeviceDX11::ReadImpl(GfxTexture* source, void* target) const
	{
		auto dxTexture = static_cast<GfxTextureDX11*>(source);
		m_DeviceContext->CopyResource(dxTexture->m_StagingTexture.Get(), dxTexture->m_Texture.Get());

		D3D11_MAPPED_SUBRESOURCE mappedTexture;
		ZeroMemory(&mappedTexture, sizeof(D3D11_MAPPED_SUBRESOURCE));
		HRESULT hr = m_DeviceContext->Map(dxTexture->m_StagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedTexture);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get texture data."));
			return;
		}
		memcpy(target, mappedTexture.pData, mappedTexture.RowPitch * dxTexture->m_Height * dxTexture->m_ArraySize);
		m_DeviceContext->Unmap(dxTexture->m_StagingTexture.Get(), 0);
	}

	void GfxDeviceDX11::ReadImpl(GfxTexture* source, void* target, const Rectangle& area) const
	{
		auto dxTexture = static_cast<GfxTextureDX11*>(source);
		m_DeviceContext->CopyResource(dxTexture->m_StagingTexture.Get(), dxTexture->m_Texture.Get());
		
		D3D11_MAPPED_SUBRESOURCE mappedTexture;
		ZeroMemory(&mappedTexture, sizeof(D3D11_MAPPED_SUBRESOURCE));
		HRESULT hr = m_DeviceContext->Map(dxTexture->m_StagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedTexture);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get texture data."));
			return;
		}
		for (int i = 0; i < area.height; i++)
		{
			size_t pixelSize = mappedTexture.RowPitch / dxTexture->m_Width;
			size_t offset = ((area.y + i) * dxTexture->m_Width + area.x) * pixelSize;
			char* copyPtr = static_cast<char*>(mappedTexture.pData) + offset;
			char* targetPtr = static_cast<char*>(target) + (area.width * pixelSize * i);
			memcpy(targetPtr, copyPtr, area.width * pixelSize);
		}
		m_DeviceContext->Unmap(dxTexture->m_StagingTexture.Get(), 0);
	}

	void GfxDeviceDX11::SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture)
	{
		SetRenderTargetImpl(renderTexture, depthStencilTexture, UINT32_MAX);
	}

	void GfxDeviceDX11::SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, const uint32_t& slice)
	{
		Clear();

		ID3D11RenderTargetView** renderTarget = nullptr;
		if (renderTexture != nullptr)
		{
			GfxTextureDX11* dxRenderTarget = static_cast<GfxTextureDX11*>(renderTexture);
			renderTarget = slice == UINT32_MAX ? dxRenderTarget->m_RenderTargetView.GetAddressOf() : dxRenderTarget->m_SlicesRenderTargetViews[slice].GetAddressOf();
			m_BindedRenderTarget = dxRenderTarget;
		}
		else
		{
			m_BindedRenderTarget = nullptr;
		}
		ID3D11DepthStencilView* depthStencil = nullptr;
		if (depthStencilTexture != nullptr)
		{
			GfxTextureDX11* dxDepthStencil = static_cast<GfxTextureDX11*>(depthStencilTexture);
			depthStencil = dxDepthStencil->m_DepthStencilView.Get();
			m_BindedDepthStencil = dxDepthStencil;
		}
		else
		{
			m_BindedDepthStencil = nullptr;
		}

		if (renderTarget == nullptr && depthStencil == nullptr)
		{
			m_DeviceContext->OMSetRenderTargets(1, m_RenderTargetView.GetAddressOf(), NULL);
		}
		else
		{
			m_DeviceContext->OMSetRenderTargets(renderTarget == nullptr ? 0 : 1, renderTarget, depthStencil);
		}
	}

	void GfxDeviceDX11::SetGlobalConstantBufferImpl(const size_t& id, GfxConstantBuffer* buffer)
	{
		auto dxConstantBuffer = static_cast<GfxConstantBufferDX11*>(buffer);
		m_BindedConstantBuffers.insert_or_assign(id, dxConstantBuffer);
		m_CurrentCrc = UINT32_MAX;
	}

	void GfxDeviceDX11::SetGlobalStructuredBufferImpl(const size_t& id, GfxStructuredBuffer* buffer)
	{
		auto dxStructuredBuffer = static_cast<GfxStructuredBufferDX11*>(buffer);
		m_BindedStructuredBuffers.insert_or_assign(id, dxStructuredBuffer);
		m_CurrentCrc = UINT32_MAX;
	}

	void GfxDeviceDX11::SetGlobalTextureImpl(const size_t& id, GfxTexture* texture)
	{
		auto dxTexture = static_cast<GfxTextureDX11*>(texture);
		for (auto it = m_BindedTextures.begin(); it < m_BindedTextures.end(); ++it)
		{
			if (it->first == id)
			{
				it->second = dxTexture;
				return;
			}
		}
		m_BindedTextures.emplace_back(std::make_pair(id, dxTexture));
		m_CurrentCrc = UINT32_MAX;
	}

	D3D11_PRIMITIVE_TOPOLOGY GetPrimitiveTopology(const Topology& topology)
	{
		switch (topology)
		{
		case Topology::Unknown:			return D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
		case Topology::PointList:		return D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
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

		const GfxRenderStateDX11 renderState = m_StateCache.GetState(operation.material, operation.passIndex);
		
		if (!renderState.isValid)
		{
			return;
		}

		ID3D11InputLayout* inputLayout = m_LayoutCache.GetLayout(renderState.dxVertexShader, operation.layout);

		if (inputLayout != m_InputLayout)
		{
			m_DeviceContext->IASetInputLayout(inputLayout);
			m_InputLayout = inputLayout;
		}
		if (renderState.vertexShader != m_RenderState.vertexShader)
		{
			m_DeviceContext->VSSetShader(renderState.vertexShader, NULL, 0);
		}
		if (renderState.geometryShader != m_RenderState.geometryShader)
		{
			m_DeviceContext->GSSetShader(renderState.geometryShader, NULL, 0);
		}
		if (renderState.pixelShader != m_RenderState.pixelShader)
		{
			m_DeviceContext->PSSetShader(renderState.pixelShader, NULL, 0);
		}
		for (uint32_t i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 8; ++i)
		{
			if (renderState.vertexShaderResourceViews[i] != m_RenderState.vertexShaderResourceViews[i])
			{
				m_DeviceContext->VSSetShaderResources(i, 1, &renderState.vertexShaderResourceViews[i]);
			}
			if (renderState.vertexSamplerStates[i] != m_RenderState.vertexSamplerStates[i])
			{
				m_DeviceContext->VSSetSamplers(i, 1, &renderState.vertexSamplerStates[i]);
			}
			if (renderState.pixelShaderResourceViews[i] != m_RenderState.pixelShaderResourceViews[i])
			{
				m_DeviceContext->PSSetShaderResources(i, 1, &renderState.pixelShaderResourceViews[i]);
			}
			if (renderState.pixelSamplerStates[i] != m_RenderState.pixelSamplerStates[i])
			{
				m_DeviceContext->PSSetSamplers(i, 1, &renderState.pixelSamplerStates[i]);
			}
		}
		for (uint32_t i = 0; i < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++i)
		{
			if (renderState.vertexConstantBuffers[i] != m_RenderState.vertexConstantBuffers[i])
			{
				m_DeviceContext->VSSetConstantBuffers(i, 1, &renderState.vertexConstantBuffers[i]);
			}
			if (renderState.geometryConstantBuffers[i] != m_RenderState.geometryConstantBuffers[i])
			{
				m_DeviceContext->GSSetConstantBuffers(i, 1, &renderState.geometryConstantBuffers[i]);
			}
			if (renderState.pixelConstantBuffers[i] != m_RenderState.pixelConstantBuffers[i])
			{
				m_DeviceContext->PSSetConstantBuffers(i, 1, &renderState.pixelConstantBuffers[i]);
			}
		}

		if (renderState.rasterizerState != m_RenderState.rasterizerState)
		{
			m_DeviceContext->RSSetState(renderState.rasterizerState);
		}
		if (renderState.depthStencilState != m_RenderState.depthStencilState)
		{
			m_DeviceContext->OMSetDepthStencilState(renderState.depthStencilState, 0);
		}
		if (renderState.blendState != m_RenderState.blendState)
		{
			const float blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
			m_DeviceContext->OMSetBlendState(renderState.blendState, blendFactor, 0xffffffff);
		}
		if (m_Topology != operation.topology)
		{
			m_Topology = operation.topology;
			m_DeviceContext->IASetPrimitiveTopology(GetPrimitiveTopology(operation.topology));
		}

		auto dxVertexBuffer = static_cast<GfxVertexBufferDX11*>(operation.vertexBuffer);
		if (dxVertexBuffer != m_VertexBuffer)
		{
			m_VertexBuffer = dxVertexBuffer;
			m_DeviceContext->IASetVertexBuffers(0, 1, dxVertexBuffer->m_Buffer.GetAddressOf(), &dxVertexBuffer->m_Stride, &dxVertexBuffer->m_Offset);
		}

		auto dxInstanceBuffer = static_cast<GfxVertexBufferDX11*>(operation.instanceBuffer);
		if (dxInstanceBuffer != m_InstanceBuffer || operation.instanceOffset != m_InstanceOffset)
		{
			m_InstanceBuffer = dxInstanceBuffer;
			m_InstanceOffset = operation.instanceOffset;
			if (dxInstanceBuffer != nullptr)
			{
				uint32_t byteOffset = m_InstanceBuffer ? m_InstanceOffset * m_InstanceBuffer->m_Stride : 0;
				m_DeviceContext->IASetVertexBuffers(1, 1, dxInstanceBuffer->m_Buffer.GetAddressOf(), &dxInstanceBuffer->m_Stride, &byteOffset);
			}
		}

		if (operation.indexBuffer == nullptr)
		{
			if (m_InstanceBuffer == nullptr)
			{
				m_DeviceContext->DrawInstanced(operation.vertexCount, m_ViewCount, 0, 0);
			}
			else
			{
				m_DeviceContext->DrawInstanced(operation.vertexCount, operation.instanceCount * m_ViewCount, 0, 0);
			}
		}
		else
		{
			auto dxIndexBuffer = static_cast<GfxIndexBufferDX11*>(operation.indexBuffer);
			if (dxIndexBuffer != m_IndexBuffer)
			{
				m_IndexBuffer = dxIndexBuffer;
				m_DeviceContext->IASetIndexBuffer(dxIndexBuffer->m_Buffer.Get(), DXGI_FORMAT_R32_UINT, 0);
			}
			if (m_InstanceBuffer == nullptr)
			{
				m_DeviceContext->DrawIndexedInstanced(operation.indexCount, m_ViewCount, operation.indexOffset, 0, 0);
			}
			else
			{
				m_DeviceContext->DrawIndexedInstanced(operation.indexCount, operation.instanceCount * m_ViewCount, operation.indexOffset, 0, 0);
			}
		}

		m_RenderState = renderState;
	}

	void GfxDeviceDX11::DispatchImpl(GfxComputeShader*& shader, const uint32_t& threadGroupsX, const uint32_t& threadGroupsY, const uint32_t& threadGroupsZ) const
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

	ID3D11Device* GfxDeviceDX11::GetDevice()
	{
		return m_Device.Get();
	}

	ID3D11DeviceContext* GfxDeviceDX11::GetDeviceContext()
	{
		return m_DeviceContext.Get();
	}

	HWND GfxDeviceDX11::GetHwnd()
	{
		return m_Hwnd;
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

		m_BindedRenderTarget = nullptr;
		m_BindedDepthStencil = nullptr;

		BB_INFO("DirectX initialized successful.");

		return true;
	}

	void GfxDeviceDX11::Clear()
	{
		// Clear SRVs to avoid binding them both into targets and inputs
		for (uint32_t i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 8; ++i)
		{
			m_RenderState.vertexShaderResourceViews[i] = nullptr;
			m_RenderState.vertexSamplerStates[i] = nullptr;
			m_RenderState.pixelShaderResourceViews[i] = nullptr;
			m_RenderState.pixelSamplerStates[i] = nullptr;
		}
		m_DeviceContext->VSSetShaderResources(0, 16, m_EmptyShaderResourceViews);
		m_DeviceContext->VSSetSamplers(0, 16, m_EmptySamplers);
		m_DeviceContext->PSSetShaderResources(0, 16, m_EmptyShaderResourceViews);
		m_DeviceContext->PSSetSamplers(0, 16, m_EmptySamplers);
	}

	ID3D11RasterizerState* GfxDeviceDX11::GetRasterizerState(const CullMode& mode)
	{
		size_t key = static_cast<size_t>(mode) | static_cast<size_t>(m_DepthBias) << 8 | *(reinterpret_cast<size_t*>(&m_SlopeDepthBias)) << 16;
		auto it = m_RasterizerStates.find(key);
		if (it != m_RasterizerStates.end())
		{
			return it->second.Get();
		}
		else
		{
			D3D11_RASTERIZER_DESC rasterizerDesc;
			ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

			rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
			rasterizerDesc.CullMode = static_cast<D3D11_CULL_MODE>(static_cast<uint32_t>(mode) + 1);
			rasterizerDesc.MultisampleEnable = true;
			rasterizerDesc.AntialiasedLineEnable = true;
			rasterizerDesc.DepthBias = m_DepthBias;
			rasterizerDesc.SlopeScaledDepthBias = m_SlopeDepthBias;

			ComPtr<ID3D11RasterizerState> state;
			HRESULT hr = m_Device->CreateRasterizerState(&rasterizerDesc, state.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create cull back rasterizer state."));
				return false;
			}
			m_RasterizerStates.insert_or_assign(key, state);
			return state.Get();
		}
	}

	D3D11_BLEND GetBlend(const BlendMode& blend)
	{
		switch (blend)
		{
		case BlendMode::One: return D3D11_BLEND::D3D11_BLEND_ONE;
		case BlendMode::Zero: return D3D11_BLEND::D3D11_BLEND_ZERO;
		case BlendMode::SrcAlpha: return D3D11_BLEND::D3D11_BLEND_SRC_ALPHA;
		case BlendMode::OneMinusSrcAlpha: return D3D11_BLEND::D3D11_BLEND_INV_SRC_ALPHA;
		default: return D3D11_BLEND::D3D11_BLEND_ONE;
		}
	}

	ID3D11BlendState* GfxDeviceDX11::GetBlendState(const BlendMode& blendSrcColor, const BlendMode& blendSrcAlpha, const BlendMode& blendDstColor, const BlendMode& blendDstAlpha)
	{
		size_t key = static_cast<size_t>(blendSrcColor) << 8 | static_cast<size_t>(blendSrcAlpha) << 16 | static_cast<size_t>(blendSrcColor) << 24 | static_cast<size_t>(blendSrcAlpha) << 32;
		auto it = m_BlendStates.find(key);
		if (it != m_BlendStates.end())
		{
			return it->second.Get();
		}
		else
		{
			D3D11_BLEND_DESC blendDesc;
			ZeroMemory(&blendDesc, sizeof(D3D11_BLEND_DESC));

			D3D11_BLEND srcColor = GetBlend(blendSrcColor);
			D3D11_BLEND srcAlpha = GetBlend(blendSrcAlpha);
			D3D11_BLEND dstColor = GetBlend(blendDstAlpha);
			D3D11_BLEND dstAlpha = GetBlend(blendDstAlpha);

			blendDesc.AlphaToCoverageEnable = false;
			blendDesc.RenderTarget[0].BlendEnable = true;
			blendDesc.RenderTarget[0].SrcBlend = srcColor;
			blendDesc.RenderTarget[0].DestBlend = dstColor;
			blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
			blendDesc.RenderTarget[0].SrcBlendAlpha = srcAlpha;
			blendDesc.RenderTarget[0].DestBlendAlpha = dstAlpha;
			blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
			blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

			ComPtr<ID3D11BlendState> state;
			HRESULT hr = m_Device->CreateBlendState(&blendDesc, state.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create blend state."));
				return state.Get();
			}
			m_BlendStates.insert_or_assign(key, state);
			return state.Get();
		}
	}

	ID3D11DepthStencilState* GfxDeviceDX11::GetDepthStencilState(const ZTest& zTest, const ZWrite& zWrite)
	{
		size_t key = static_cast<size_t>(zTest) << 8 | static_cast<size_t>(zWrite) << 16;
		auto it = m_DepthStencilStates.find(key);
		if (it != m_DepthStencilStates.end())
		{
			return it->second.Get();
		}
		else
		{
			D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
			ZeroMemory(&depthStencilDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));

			depthStencilDesc.DepthEnable = true;
			depthStencilDesc.DepthWriteMask = zWrite == ZWrite::On ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
			depthStencilDesc.DepthFunc = (D3D11_COMPARISON_FUNC)(static_cast<uint32_t>(zTest) + 1);

			ComPtr<ID3D11DepthStencilState> state;
			HRESULT hr = m_Device->CreateDepthStencilState(&depthStencilDesc, state.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil state."));
				return state.Get();
			}
			m_DepthStencilStates.insert_or_assign(key, state);
			return state.Get();
		}
	}

	const uint32_t& GfxDeviceDX11::GetCRC()
	{
		if (m_CurrentCrc == UINT32_MAX)
		{
			m_CurrentCrc = 0;
			for (auto& pair : m_BindedTextures)
			{
				m_CurrentCrc = CRCHelper::Calculate(&pair, sizeof(size_t), m_CurrentCrc);
			}
			for (auto& pair : m_BindedConstantBuffers)
			{
				m_CurrentCrc = CRCHelper::Calculate(&pair, sizeof(size_t), m_CurrentCrc);
			}
			for (auto& pair : m_BindedStructuredBuffers)
			{
				m_CurrentCrc = CRCHelper::Calculate(&pair, sizeof(size_t), m_CurrentCrc);
			}
		}
		return m_CurrentCrc;
	}
}