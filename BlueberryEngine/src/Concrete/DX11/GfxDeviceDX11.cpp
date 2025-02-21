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

		// Clear
		ID3D11ShaderResourceView* emptySRV[1] = { nullptr };
		ID3D11SamplerState* emptySampler[1] = { nullptr };
		m_DeviceContext->PSSetShaderResources(0, 1, emptySRV);
		m_DeviceContext->PSSetSamplers(0, 1, emptySampler);

		m_BindedConstantBuffers.clear();
		m_BindedStructuredBuffers.clear();
		m_BindedTextures.clear();
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

	bool GfxDeviceDX11::CreateVertexBufferImpl(const VertexLayout& layout, const uint32_t& vertexCount, GfxVertexBuffer*& buffer)
	{
		auto dxBuffer = new GfxVertexBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(layout, vertexCount))
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
		m_DeviceContext->PSSetShaderResources(0, 16, m_ShaderResourceViews);
		m_DeviceContext->PSSetSamplers(0, 16, m_Samplers);

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
		m_CurrentCrc = 0;
	}

	void GfxDeviceDX11::SetGlobalStructuredBufferImpl(const size_t& id, GfxStructuredBuffer* buffer)
	{
		auto dxStructuredBuffer = static_cast<GfxStructuredBufferDX11*>(buffer);
		m_BindedStructuredBuffers.insert_or_assign(id, dxStructuredBuffer);
		m_CurrentCrc = 0;
	}

	void GfxDeviceDX11::SetGlobalTextureImpl(const size_t& id, GfxTexture* texture)
	{
		auto dxTexture = static_cast<GfxTextureDX11*>(texture);
		m_BindedTextures.insert_or_assign(id, dxTexture);
		m_CurrentCrc = 0;
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

		GfxRenderState* renderState = operation.renderState;
		ObjectId materialId = operation.materialId;
		uint32_t materialCRC = operation.materialCRC;
		SetCullMode(renderState->cullMode);
		SetBlendMode(renderState->blendSrcColor, renderState->blendSrcAlpha, renderState->blendDstColor, renderState->blendDstAlpha);
		SetZTestAndZWrite(renderState->zTest, renderState->zWrite);
		SetTopology(operation.topology);

		// TODO check if shader variant/material/mesh is the same to skip some bindings

		// Does not work when texture is switched for example in IconRenderer
		{
			auto dxVertexShader = static_cast<GfxVertexShaderDX11*>(renderState->vertexShader);
			if (dxVertexShader != m_VertexShader)
			{
				m_VertexShader = dxVertexShader;

				m_DeviceContext->IASetInputLayout(dxVertexShader->m_InputLayout.Get());
				m_DeviceContext->VSSetShader(dxVertexShader->m_Shader.Get(), NULL, 0);
			}

			auto dxGeometryShader = static_cast<GfxGeometryShaderDX11*>(renderState->geometryShader);
			if (dxGeometryShader != m_GeometryShader)
			{
				m_GeometryShader = dxGeometryShader;
				
				if (dxGeometryShader == nullptr)
				{
					m_DeviceContext->GSSetShader(NULL, NULL, 0);
				}
				else
				{
					m_DeviceContext->GSSetShader(dxGeometryShader->m_Shader.Get(), NULL, 0);
				}
			}
			
			auto dxFragmentShader = static_cast<GfxFragmentShaderDX11*>(renderState->fragmentShader);
			if (dxFragmentShader != m_FragmentShader)
			{
				m_FragmentShader = dxFragmentShader;

				m_DeviceContext->PSSetShader(dxFragmentShader->m_Shader.Get(), NULL, 0);
			}

			if (materialId != m_MaterialId || operation.materialCRC != m_MaterialCrc || GetCRC() != m_GlobalCrc || renderState != m_RenderState)
			{
				m_MaterialId = materialId;
				m_RenderState = renderState;
				m_MaterialCrc = materialCRC;
				m_GlobalCrc = m_CurrentCrc;

				// Bind vertex constant buffers
				for (auto it = dxVertexShader->m_ConstantBufferSlots.begin(); it != dxVertexShader->m_ConstantBufferSlots.end(); it++)
				{
					auto pair = m_BindedConstantBuffers.find(it->first);
					if (pair != m_BindedConstantBuffers.end())
					{
						m_ConstantBuffers[it->second] = pair->second->m_Buffer.Get();
					}
				}

				m_DeviceContext->VSSetConstantBuffers(0, 8, m_ConstantBuffers);

				std::fill_n(m_ConstantBuffers, 8, nullptr);
				std::fill_n(m_ShaderResourceViews, 16, nullptr);

				// Bind vertex structured buffers
				for (auto it = dxVertexShader->m_StructuredBufferSlots.begin(); it != dxVertexShader->m_StructuredBufferSlots.end(); it++)
				{
					auto pair = m_BindedStructuredBuffers.find(it->first);
					if (pair != m_BindedStructuredBuffers.end())
					{
						uint32_t bufferSlotIndex = it->second.first;
						uint32_t shaderResourceViewSlotIndex = it->second.second;
						auto dxBuffer = pair->second;
						//m_ConstantBuffers[bufferSlotIndex] = dxBuffer->m_Buffer.Get();
						m_ShaderResourceViews[shaderResourceViewSlotIndex] = dxBuffer->m_ShaderResourceView.Get();
					}
				}
				
				m_DeviceContext->VSSetShaderResources(0, 16, m_ShaderResourceViews);

				if (dxGeometryShader != nullptr)
				{
					// Bind geometry constant buffers
					for (auto it = dxGeometryShader->m_ConstantBufferSlots.begin(); it != dxGeometryShader->m_ConstantBufferSlots.end(); it++)
					{
						auto pair = m_BindedConstantBuffers.find(it->first);
						if (pair != m_BindedConstantBuffers.end())
						{
							m_ConstantBuffers[it->second] = pair->second->m_Buffer.Get();
						}
					}

					m_DeviceContext->GSSetConstantBuffers(0, 8, m_ConstantBuffers);
				}

				// Bind fragment constant buffers
				for (auto it = dxFragmentShader->m_ConstantBufferSlots.begin(); it != dxFragmentShader->m_ConstantBufferSlots.end(); it++)
				{
					auto pair = m_BindedConstantBuffers.find(it->first);
					if (pair != m_BindedConstantBuffers.end())
					{
						m_ConstantBuffers[it->second] = pair->second->m_Buffer.Get();
					}
				}

				m_DeviceContext->PSSetConstantBuffers(0, 8, m_ConstantBuffers);

				std::fill_n(m_ConstantBuffers, 8, nullptr);

				// Bind fragment material textures
				for (int i = 0; i < renderState->fragmentTextureCount; i++)
				{
					GfxRenderState::TextureInfo info = renderState->fragmentTextures[i];
					auto dxTexture = static_cast<GfxTextureDX11*>(info.Get());
					m_ShaderResourceViews[info.textureSlot] = dxTexture->m_ResourceView.Get();
					if (info.samplerSlot != -1)
					{
						m_Samplers[info.samplerSlot] = dxTexture->m_SamplerState.Get();
					}
				}

				// Bind fragment global textures
				for (auto it = dxFragmentShader->m_TextureSlots.begin(); it != dxFragmentShader->m_TextureSlots.end(); it++)
				{
					auto pair = m_BindedTextures.find(it->first);
					if (pair != m_BindedTextures.end())
					{
						uint32_t textureSlotIndex = it->second.first;
						uint32_t samplerSlotIndex = it->second.second;
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

				std::fill_n(m_ShaderResourceViews, 16, nullptr);
				std::fill_n(m_Samplers, 16, nullptr);
			}
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

		// None
		{
			D3D11_RASTERIZER_DESC rasterizerDesc;
			ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

			rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
			rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE;
			rasterizerDesc.MultisampleEnable = true;
			rasterizerDesc.AntialiasedLineEnable = true;
			rasterizerDesc.DepthBias = 8;
			rasterizerDesc.SlopeScaledDepthBias = 2.13f;

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
			rasterizerDesc.DepthBias = 8;
			rasterizerDesc.SlopeScaledDepthBias = 2.13f;

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
			rasterizerDesc.DepthBias = 8;
			rasterizerDesc.SlopeScaledDepthBias = 2.13f;

			hr = m_Device->CreateRasterizerState(&rasterizerDesc, m_CullBackRasterizerState.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create cull back rasterizer state."));
				return false;
			}
		}

		SetCullMode(CullMode::None);
		SetBlendMode(BlendMode::One, BlendMode::Zero, BlendMode::One, BlendMode::Zero);
		SetZTestAndZWrite(ZTest::LessEqual, ZWrite::On);

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

	void GfxDeviceDX11::SetBlendMode(const BlendMode& blendSrcColor, const BlendMode& blendSrcAlpha, const BlendMode& blendDstColor, const BlendMode& blendDstAlpha)
	{
		size_t key = static_cast<size_t>(blendSrcColor) << 8 | static_cast<size_t>(blendSrcAlpha) << 16 | static_cast<size_t>(blendSrcColor) << 24 | static_cast<size_t>(blendSrcAlpha) << 32;
		if (key == m_BlendKey)
		{
			return;
		}
		m_BlendKey = key;

		ComPtr<ID3D11BlendState> state;
		auto it = m_BlendStates.find(key);
		if (it != m_BlendStates.end())
		{
			state = it->second;
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

			HRESULT hr = m_Device->CreateBlendState(&blendDesc, state.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create blend state."));
				return;
			}
			m_BlendStates.insert_or_assign(key, state);
		}
		const float blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
		m_DeviceContext->OMSetBlendState(state.Get(), blendFactor, 0xffffffff);
	}

	void GfxDeviceDX11::SetZTestAndZWrite(const ZTest& zTest, const ZWrite& zWrite)
	{
		size_t key = static_cast<size_t>(zTest) << 8 | static_cast<size_t>(zWrite) << 16;
		if (key == m_ZTestZWriteKey)
		{
			return;
		}
		m_ZTestZWriteKey = key;

		ComPtr<ID3D11DepthStencilState> state;
		auto it = m_DepthStencilStates.find(key);
		if (it != m_DepthStencilStates.end())
		{
			state = it->second;
		}
		else
		{
			D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
			ZeroMemory(&depthStencilDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));

			depthStencilDesc.DepthEnable = true;
			depthStencilDesc.DepthWriteMask = zWrite == ZWrite::On ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
			depthStencilDesc.DepthFunc = (D3D11_COMPARISON_FUNC)(static_cast<uint32_t>(zTest) + 1);

			HRESULT hr = m_Device->CreateDepthStencilState(&depthStencilDesc, state.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil state."));
				return;
			}
			m_DepthStencilStates.insert_or_assign(key, state);
		}
		m_DeviceContext->OMSetDepthStencilState(state.Get(), 0);
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

	const uint32_t& GfxDeviceDX11::GetCRC()
	{
		if (m_CurrentCrc == 0)
		{
			for (auto& pair : m_BindedTextures)
			{
				m_CurrentCrc = CRCHelper::Calculate(&pair, sizeof(std::pair<size_t, GfxTextureDX11*>), m_CurrentCrc);
			}
			for (auto& pair : m_BindedConstantBuffers)
			{
				m_CurrentCrc = CRCHelper::Calculate(&pair, sizeof(std::pair<size_t, GfxConstantBufferDX11*>), m_CurrentCrc);
			}
			for (auto& pair : m_BindedStructuredBuffers)
			{
				m_CurrentCrc = CRCHelper::Calculate(&pair, sizeof(std::pair<size_t, GfxStructuredBufferDX11*>), m_CurrentCrc);
			}
		}
		return m_CurrentCrc;
	}
}