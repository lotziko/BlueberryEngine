#include "GfxDeviceDX11.h"

#include "GfxShaderDX11.h"
#include "GfxComputeShaderDX11.h"
#include "GfxBufferDX11.h"
#include "GfxTextureDX11.h"
#include "ImGuiRendererDX11.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Tools\CRCHelper.h"
#include "..\Windows\WindowsHelper.h"

#include "Blueberry\Logging\Profiler.h"

namespace Blueberry
{
	#define BUFFER_COUNT 2

	GfxDeviceDX11::~GfxDeviceDX11()
	{
		m_SwapChain = nullptr;
		m_BackbufferRenderTargetView = nullptr;
		m_DeviceContext = nullptr;
		m_FrameLatencyWaitHandle = nullptr;
	}

	bool GfxDeviceDX11::InitializeImpl(int width, int height, void* data)
	{
		if (!InitializeDirectX(*(static_cast<HWND*>(data)), width, height))
			return false;

		m_StateCache = GfxRenderStateCacheDX11(this);
		m_ComputeStateCache = GfxComputeRenderStateCacheDX11(this);
		m_BindedTextures.reserve(64);

		return true;
	}

	void GfxDeviceDX11::ClearColorImpl(const Color& color)
	{
		if (m_BindedRenderTargets[0] == nullptr)
		{
			m_DeviceContext->ClearRenderTargetView(m_BackbufferRenderTargetView.Get(), color);
		}
		else
		{
			for (uint32_t i = 0; i < BINDED_TARGET_COUNT; ++i)
			{
				GfxTextureDX11* renderTarget = m_BindedRenderTargets[i];
				if (renderTarget != nullptr)
				{
					m_DeviceContext->ClearRenderTargetView(renderTarget->GetRenderTargetView(), color);
				}
			}
		}
	}

	void GfxDeviceDX11::ClearDepthImpl(float depth)
	{
		if (m_BindedDepthStencil != nullptr)
		{
			m_DeviceContext->ClearDepthStencilView(m_BindedDepthStencil->GetDepthStencilView(), D3D11_CLEAR_DEPTH, depth, 0);
		}
	}

	void GfxDeviceDX11::WaitForFrameImpl()
	{
		if (BUFFER_COUNT > 1)
		{
			WaitForSingleObjectEx(m_FrameLatencyWaitHandle, 16, TRUE);
		}
	}

	void GfxDeviceDX11::SwapBuffersImpl()
	{
		m_SwapChain->Present(1, 0);
		Clear();
	}

	void GfxDeviceDX11::SetViewportImpl(int x, int y, int width, int height)
	{
		D3D11_VIEWPORT viewport = {};
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
			D3D11_RECT rect = {};
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
		m_BackbufferRenderTargetView->Release();

		HRESULT hr = m_SwapChain->ResizeBuffers(BUFFER_COUNT, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, BUFFER_COUNT > 1 ? DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT : 0);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "ResizeBuffers failed."));
			return;
		}

		ComPtr<ID3D11Texture2D> backBuffer;
		hr = m_SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backBuffer.GetAddressOf()));
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "GetBuffer failed."));
			return;
		}

		hr = m_Device->CreateRenderTargetView(backBuffer.Get(), NULL, m_BackbufferRenderTargetView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
			return;
		}

		m_DeviceContext->OMSetRenderTargets(1, m_BackbufferRenderTargetView.GetAddressOf(), NULL);
	}

	uint32_t GfxDeviceDX11::GetViewCountImpl()
	{
		return m_ViewCount;
	}

	void GfxDeviceDX11::SetViewCountImpl(uint32_t count)
	{
		m_ViewCount = count;
	}

	void GfxDeviceDX11::SetDepthBiasImpl(uint32_t bias, float slopeBias)
	{
		m_DepthBias = bias;
		m_SlopeDepthBias = slopeBias;
	}

	bool GfxDeviceDX11::CreateVertexShaderImpl(const ByteData& vertexData, GfxVertexShader*& shader)
	{
		auto dxShader = new GfxVertexShaderDX11();
		if (!dxShader->Initialize(m_Device.Get(), vertexData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateGeometryShaderImpl(const ByteData& geometryData, GfxGeometryShader*& shader)
	{
		auto dxShader = new GfxGeometryShaderDX11();
		if (!dxShader->Initialize(m_Device.Get(), geometryData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateFragmentShaderImpl(const ByteData& fragmentData, GfxFragmentShader*& shader)
	{
		auto dxShader = new GfxFragmentShaderDX11();
		if (!dxShader->Initialize(m_Device.Get(), fragmentData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateComputeShaderImpl(const ByteData& computeData, GfxComputeShader*& shader)
	{
		auto dxShader = new GfxComputeShaderDX11();
		if (!dxShader->Initialize(m_Device.Get(), computeData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX11::CreateRayTracingShaderImpl(const ByteData& rayTracingData, GfxRayTracingShader*& shader)
	{
		return false;
	}

	bool GfxDeviceDX11::CreateBufferImpl(const BufferProperties& properties, GfxBuffer*& buffer)
	{
		auto dxBuffer = new GfxBufferDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxBuffer->Initialize(properties))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX11::CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture)
	{
		GfxTextureDX11* dxTexture = new GfxTextureDX11(m_Device.Get(), m_DeviceContext.Get());
		if (!dxTexture->Initialize(properties))
		{
			return false;
		}
		texture = dxTexture;
		return true;
	}

	bool GfxDeviceDX11::CreateBottomLevelAccelerationStructureImpl(const BottomLevelAccelerationStructureProperties& properties, GfxBottomLevelAccelerationStructure*& accelerationStructure)
	{
		return false;
	}

	bool GfxDeviceDX11::CreateTopLevelAccelerationStructureImpl(GfxTopLevelAccelerationStructure*& accelerationStructure)
	{
		return false;
	}

	void GfxDeviceDX11::CopyImpl(GfxTexture* source, GfxTexture* target)
	{
		m_DeviceContext->CopyResource(static_cast<GfxTextureDX11*>(target)->GetResource(), static_cast<GfxTextureDX11*>(source)->GetResource());
	}

	void GfxDeviceDX11::CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area)
	{
		D3D11_BOX src = {};
		src.left = static_cast<UINT>(area.x);
		src.top = static_cast<UINT>(area.y);
		src.right = static_cast<UINT>(area.x + area.width);
		src.bottom = static_cast<UINT>(area.y + area.height);
		src.front = 0;
		src.back = 1;

		m_DeviceContext->CopySubresourceRegion(static_cast<GfxTextureDX11*>(target)->GetResource(), 0, 0, 0, 0, static_cast<GfxTextureDX11*>(source)->GetResource(), 0, &src);
	}

	void GfxDeviceDX11::CopyImpl(GfxTexture* source, GfxTexture* target, const Vector2Int& offset, const Rectangle& area)
	{
		D3D11_BOX src = {};
		src.left = static_cast<UINT>(area.x);
		src.top = static_cast<UINT>(area.y);
		src.right = static_cast<UINT>(area.x + area.width);
		src.bottom = static_cast<UINT>(area.y + area.height);
		src.front = 0;
		src.back = 1;

		m_DeviceContext->CopySubresourceRegion(static_cast<GfxTextureDX11*>(target)->GetResource(), static_cast<UINT>(offset.x), static_cast<UINT>(offset.y), 0, 0, static_cast<GfxTextureDX11*>(source)->GetResource(), 0, &src);
	}

	void GfxDeviceDX11::CopyImpl(GfxTexture* source, GfxTexture* target, uint32_t sourceSlice, uint32_t targetSlice, uint32_t mipLevel)
	{
		GfxTextureDX11* dxSource = static_cast<GfxTextureDX11*>(source);
		GfxTextureDX11* dxTarget = static_cast<GfxTextureDX11*>(target);

		UINT sourceSubresource = D3D11CalcSubresource(mipLevel, sourceSlice, dxSource->GetMipLevels());
		UINT targetSubresource = D3D11CalcSubresource(mipLevel, targetSlice, dxTarget->GetMipLevels());

		m_DeviceContext->CopySubresourceRegion(dxTarget->GetResource(), targetSubresource, 0, 0, 0, dxSource->GetResource(), sourceSubresource, NULL);
	}

	void GfxDeviceDX11::SetRenderTargetImpl(GfxTexture** renderTextures, uint32_t renderTexturesCount, GfxTexture* depthStencilTexture, uint32_t arraySlice, uint32_t mipLevel)
	{
		Clear();

		ID3D11RenderTargetView* renderTargets[BINDED_TARGET_COUNT] = {};
		if (renderTexturesCount > 0)
		{
			for (uint32_t i = 0; i < renderTexturesCount; ++i)
			{
				GfxTextureDX11* dxRenderTarget = static_cast<GfxTextureDX11*>(renderTextures[i]);

				if (arraySlice || mipLevel)
				{
					renderTargets[i] = dxRenderTarget->GetRenderTargetView(arraySlice, mipLevel);
				}
				else
				{
					renderTargets[i] = dxRenderTarget->GetRenderTargetView();
				}

				m_BindedRenderTargets[i] = dxRenderTarget;
			}
		}
		else
		{
			m_BindedRenderTargets[0] = nullptr;
		}
		ID3D11DepthStencilView* depthStencil = nullptr;
		if (depthStencilTexture != nullptr)
		{
			GfxTextureDX11* dxDepthStencil = static_cast<GfxTextureDX11*>(depthStencilTexture);
			depthStencil = dxDepthStencil->GetDepthStencilView();
			m_BindedDepthStencil = dxDepthStencil;
		}
		else
		{
			m_BindedDepthStencil = nullptr;
		}

		if (renderTargets[0] == nullptr && depthStencil == nullptr)
		{
			m_DeviceContext->OMSetRenderTargets(1, m_BackbufferRenderTargetView.GetAddressOf(), NULL);
		}
		else
		{
			m_DeviceContext->OMSetRenderTargets(renderTexturesCount, renderTargets, depthStencil);
		}
	}

	void GfxDeviceDX11::SetGlobalBufferImpl(size_t id, GfxBuffer* buffer)
	{
		auto dxBuffer = static_cast<GfxBufferDX11*>(buffer);
		for (auto& bindedBuffer : m_BindedBuffers)
		{
			if (bindedBuffer.id == id)
			{
				bindedBuffer.index = dxBuffer->GetIndex();
				return;
			}
		}
		m_BindedBuffers.push_back({ id, dxBuffer->GetIndex() });
	}

	void GfxDeviceDX11::SetGlobalTextureImpl(size_t id, GfxTexture* texture, uint32_t mip)
	{
		auto dxTexture = static_cast<GfxTextureDX11*>(texture);
		for (auto& bindedTexture : m_BindedTextures)
		{
			if (bindedTexture.id == id)
			{
				bindedTexture.index = dxTexture->GetIndex();
				bindedTexture.mip = mip;
				return;
			}
		}
		m_BindedTextures.push_back({ id, dxTexture->GetIndex(), mip });
	}

	D3D11_PRIMITIVE_TOPOLOGY GetPrimitiveTopologyD3D11(const Topology& topology)
	{
		switch (topology)
		{
		case Topology::Unknown: return D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
		case Topology::PointList: return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
		case Topology::LineList: return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
		case Topology::LineStrip: return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
		case Topology::TriangleList: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		default: return D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
		}
	}

	void GfxDeviceDX11::DrawImpl(const GfxDrawingOperation& operation)
	{
		if (!operation.IsValid())
		{
			return;
		}

		const GfxRenderStateDX11 renderState = m_StateCache.GetState(operation.material, operation.passId, operation.layout, m_DepthBias, m_SlopeDepthBias, operation.isCounterClockwise, operation.isSolid);
		
		if (!renderState.isValid)
		{
			return;
		}

		if (renderState.inputLayout != m_InputLayout)
		{
			m_DeviceContext->IASetInputLayout(renderState.inputLayout);
			m_InputLayout = renderState.inputLayout;
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
		for (uint32_t i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 4; ++i)
		{
			// Maybe bind all at once?
			if (renderState.vertexShaderResourceViews[i] != m_RenderState.vertexShaderResourceViews[i])
			{
				m_DeviceContext->VSSetShaderResources(i, 1, renderState.vertexShaderResourceViews + i);
			}
			if (renderState.pixelShaderResourceViews[i] != m_RenderState.pixelShaderResourceViews[i])
			{
				m_DeviceContext->PSSetShaderResources(i, 1, renderState.pixelShaderResourceViews + i);
			}
		}
		for (uint32_t i = 0; i < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT; ++i)
		{
			if (renderState.vertexSamplerStates[i] != m_RenderState.vertexSamplerStates[i])
			{
				m_DeviceContext->VSSetSamplers(i, 1, renderState.vertexSamplerStates + i);
			}
			if (renderState.pixelSamplerStates[i] != m_RenderState.pixelSamplerStates[i])
			{
				m_DeviceContext->PSSetSamplers(i, 1, renderState.pixelSamplerStates + i);
			}
		}
		for (uint32_t i = 0; i < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++i)
		{
			if (renderState.vertexConstantBuffers[i] != m_RenderState.vertexConstantBuffers[i])
			{
				m_DeviceContext->VSSetConstantBuffers(i, 1, renderState.vertexConstantBuffers + i);
			}
			if (renderState.geometryConstantBuffers[i] != m_RenderState.geometryConstantBuffers[i])
			{
				m_DeviceContext->GSSetConstantBuffers(i, 1, renderState.geometryConstantBuffers + i);
			}
			if (renderState.pixelConstantBuffers[i] != m_RenderState.pixelConstantBuffers[i])
			{
				m_DeviceContext->PSSetConstantBuffers(i, 1, renderState.pixelConstantBuffers + i);
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
			m_DeviceContext->IASetPrimitiveTopology(GetPrimitiveTopologyD3D11(operation.topology));
		}

		auto dxVertexBuffer = static_cast<GfxBufferDX11*>(operation.vertexBuffer);
		if (dxVertexBuffer != m_VertexBuffer)
		{
			ID3D11Buffer* buffer = dxVertexBuffer->GetBuffer();
			uint32_t elementSize = dxVertexBuffer->GetElementSize();
			uint32_t byteOffset = 0;
			m_VertexBuffer = dxVertexBuffer;
			m_DeviceContext->IASetVertexBuffers(0, 1, &buffer, &elementSize, &byteOffset);
		}

		auto dxInstanceBuffer = static_cast<GfxBufferDX11*>(operation.instanceBuffer);
		if (dxInstanceBuffer != m_InstanceBuffer || operation.instanceOffset != m_InstanceOffset)
		{
			m_InstanceBuffer = dxInstanceBuffer;
			m_InstanceOffset = operation.instanceOffset;
			if (dxInstanceBuffer != nullptr)
			{
				ID3D11Buffer* buffer = dxInstanceBuffer->GetBuffer();
				uint32_t elementSize = dxInstanceBuffer->GetElementSize();
				uint32_t byteOffset = m_InstanceBuffer ? m_InstanceOffset * elementSize : 0;
				m_DeviceContext->IASetVertexBuffers(1, 1, &buffer, &elementSize, &byteOffset);
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
			auto dxIndexBuffer = static_cast<GfxBufferDX11*>(operation.indexBuffer);
			if (dxIndexBuffer != m_IndexBuffer)
			{
				m_IndexBuffer = dxIndexBuffer;
				m_DeviceContext->IASetIndexBuffer(dxIndexBuffer->GetBuffer(), DXGI_FORMAT_R32_UINT, 0);
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

	void GfxDeviceDX11::DispatchImpl(ComputeShader* shader, uint32_t kernelIndex, uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ)
	{
		const GfxComputeRenderStateDX11 renderState = m_ComputeStateCache.GetRenderState(shader, kernelIndex);

		if (!renderState.isValid)
		{
			return;
		}

		if (renderState.computeShader != m_ComputeRenderState.computeShader)
		{
			m_DeviceContext->CSSetShader(renderState.computeShader, NULL, 0);
		}
		
		if (renderState.constantBuffersCount > 0)
		{
			m_DeviceContext->CSSetConstantBuffers(0, renderState.constantBuffersCount, renderState.constantBuffers);
		}
		if (renderState.shaderResourceViewsCount > 0)
		{
			m_DeviceContext->CSSetShaderResources(0, renderState.shaderResourceViewsCount, renderState.shaderResourceViews);
		}
		if (renderState.unorderedAccessViewsCount > 0)
		{
			m_DeviceContext->CSSetUnorderedAccessViews(0, renderState.unorderedAccessViewsCount, renderState.unorderedAccessViews, nullptr);
		}
		if (renderState.samplerStatesCount > 0)
		{
			m_DeviceContext->CSSetSamplers(0, renderState.samplerStatesCount, renderState.samplerStates);
		}
		
		m_DeviceContext->Dispatch(threadGroupsX, threadGroupsY, threadGroupsZ);
		m_DeviceContext->CSSetShaderResources(0, 16, m_EmptyShaderResourceViews);
		m_DeviceContext->CSSetSamplers(0, 16, m_EmptySamplers);
		m_DeviceContext->CSSetUnorderedAccessViews(0, 8, m_EmptyUnorderedAccessViews, NULL);
		m_ComputeRenderState = renderState;
	}

	void GfxDeviceDX11::DispatchRaysImpl(RayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, uint32_t width, uint32_t height, uint32_t depth)
	{
	}

	Matrix GfxDeviceDX11::GetGPUMatrixImpl(const Matrix& matrix) const
	{
		return matrix;
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

		ComPtr<IDXGIFactory6> dxgiFactory;
		HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating dxgi factory."));
			return false;
		}

		List<IDXGIAdapter*> adapters;
		IDXGIAdapter1* adapter = nullptr;

		for (UINT i = 0; dxgiFactory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i)
		{
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			{
				continue;
			}

			break;
		}

		DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
		swapChainDesc.BufferDesc.Width = width;
		swapChainDesc.BufferDesc.Height = height;
		swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
		swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
		swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
		swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

		swapChainDesc.SampleDesc.Count = 1;
		swapChainDesc.SampleDesc.Quality = 0;

		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.BufferCount = BUFFER_COUNT;
		swapChainDesc.OutputWindow = hwnd;
		swapChainDesc.Windowed = TRUE;
		swapChainDesc.SwapEffect = BUFFER_COUNT > 1 ? DXGI_SWAP_EFFECT_FLIP_DISCARD : DXGI_SWAP_EFFECT_DISCARD;
		swapChainDesc.Flags = (BUFFER_COUNT > 1 ? DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT : 0) | DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

		hr = D3D11CreateDeviceAndSwapChain(
			adapter,
			adapter == NULL ? D3D_DRIVER_TYPE_HARDWARE : D3D_DRIVER_TYPE_UNKNOWN, //hardware driver
			NULL, //software driver
			0, //no flags	// D3D11_CREATE_DEVICE_DEBUG does not work in runtime
			NULL, //feature levels
			0, //no feature levels
			D3D11_SDK_VERSION,
			&swapChainDesc, //swapchain description
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

		if (BUFFER_COUNT > 1)
		{
			ComPtr<IDXGISwapChain2> swapChain2;
			hr = m_SwapChain->QueryInterface(__uuidof(IDXGISwapChain2), (void**)&swapChain2);
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating swapchain."));
				return false;
			}
			swapChain2->SetMaximumFrameLatency(1);

			m_FrameLatencyWaitHandle = swapChain2->GetFrameLatencyWaitableObject();
		}
		
		ComPtr<ID3D11Texture2D> backBuffer;
		hr = m_SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backBuffer.GetAddressOf()));
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "GetBuffer failed."));
			return false;
		}

		hr = m_Device->CreateRenderTargetView(backBuffer.Get(), NULL, m_BackbufferRenderTargetView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create render target view."));
			return false;
		}

		m_DeviceContext->OMSetRenderTargets(1, m_BackbufferRenderTargetView.GetAddressOf(), NULL);

		m_BindedRenderTargets[0] = nullptr;
		m_BindedDepthStencil = nullptr;

		BB_INFO("DirectX initialized successful.");

		return true;
	}

	void GfxDeviceDX11::Clear()
	{
		// Clear SRVs to avoid binding them both into targets and inputs
		for (uint32_t i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT / 4; ++i)
		{
			if (m_RenderState.vertexShaderResourceViews[i] != nullptr)
			{
				m_RenderState.vertexShaderResourceViews[i] = {};
				m_DeviceContext->VSSetShaderResources(i, 1, m_EmptyShaderResourceViews);
			}
			if (m_RenderState.pixelShaderResourceViews[i] != nullptr)
			{
				m_RenderState.pixelShaderResourceViews[i] = {};
				m_DeviceContext->PSSetShaderResources(i, 1, m_EmptyShaderResourceViews);
			}
		}
		for (uint32_t i = 0; i < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT; ++i)
		{
			if (m_RenderState.vertexSamplerStates[i] != nullptr)
			{
				m_RenderState.vertexSamplerStates[i] = {};
				m_DeviceContext->VSSetSamplers(i, 1, m_EmptySamplers);
			}
			if (m_RenderState.pixelSamplerStates[i] != nullptr)
			{
				m_RenderState.pixelSamplerStates[i] = {};
				m_DeviceContext->PSSetSamplers(i, 1, m_EmptySamplers);
			}
		}
	}

	ID3D11RasterizerState* GfxDeviceDX11::GetRasterizerState(CullMode mode, uint32_t depthBias, float slopeDepthBias, bool isCounterClockwise, bool isSolid)
	{
		size_t key = static_cast<size_t>(mode) | static_cast<size_t>(depthBias) << 8 | *(reinterpret_cast<size_t*>(&slopeDepthBias)) << 16 | (isCounterClockwise ? 1ull : 0ull) << 24 | (isSolid ? 1ull : 0ull) << 25;
		for (auto& pair : m_RasterizerStates)
		{
			if (pair.first == key)
			{
				return pair.second.Get();
			}
		}

		D3D11_RASTERIZER_DESC rasterizerDesc = {};
		rasterizerDesc.FillMode = isSolid ? D3D11_FILL_SOLID : D3D11_FILL_WIREFRAME;
		rasterizerDesc.CullMode = static_cast<D3D11_CULL_MODE>(static_cast<uint32_t>(mode) + 1);
		rasterizerDesc.FrontCounterClockwise = isCounterClockwise;
		rasterizerDesc.MultisampleEnable = true;
		rasterizerDesc.AntialiasedLineEnable = true;
		rasterizerDesc.DepthBias = depthBias;
		rasterizerDesc.SlopeScaledDepthBias = slopeDepthBias;

		ComPtr<ID3D11RasterizerState> state;
		HRESULT hr = m_Device->CreateRasterizerState(&rasterizerDesc, state.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create cull back rasterizer state."));
			return false;
		}
		m_RasterizerStates.push_back(std::make_pair(key, state));
		return state.Get();
	}

	D3D11_BLEND GetBlend(BlendMode blend)
	{
		switch (blend)
		{
		case BlendMode::One: return D3D11_BLEND_ONE;
		case BlendMode::Zero: return D3D11_BLEND_ZERO;
		case BlendMode::SrcAlpha: return D3D11_BLEND_SRC_ALPHA;
		case BlendMode::OneMinusSrcAlpha: return D3D11_BLEND_INV_SRC_ALPHA;
		default: return D3D11_BLEND_ONE;
		}
	}

	ID3D11BlendState* GfxDeviceDX11::GetBlendState(BlendMode blendSrcColor, BlendMode blendSrcAlpha, BlendMode blendDstColor, BlendMode blendDstAlpha)
	{
		size_t key = static_cast<size_t>(blendSrcColor) << 8 | static_cast<size_t>(blendSrcAlpha) << 16 | static_cast<size_t>(blendSrcColor) << 24 | static_cast<size_t>(blendSrcAlpha) << 32;
		for (auto& pair : m_BlendStates)
		{
			if (pair.first == key)
			{
				return pair.second.Get();
			}
		}

		D3D11_BLEND srcColor = GetBlend(blendSrcColor);
		D3D11_BLEND srcAlpha = GetBlend(blendSrcAlpha);
		D3D11_BLEND dstColor = GetBlend(blendDstColor);
		D3D11_BLEND dstAlpha = GetBlend(blendDstAlpha);

		D3D11_BLEND_DESC blendDesc = {};
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
		m_BlendStates.push_back(std::make_pair(key, state));
		return state.Get();
	}

	ID3D11DepthStencilState* GfxDeviceDX11::GetDepthStencilState(ZTest zTest, ZWrite zWrite)
	{
		size_t key = static_cast<size_t>(zTest) << 8 | static_cast<size_t>(zWrite) << 16;
		for (auto& pair : m_DepthStencilStates)
		{
			if (pair.first == key)
			{
				return pair.second.Get();
			}
		}
		
		D3D11_DEPTH_STENCIL_DESC depthStencilDesc = {};
		depthStencilDesc.DepthEnable = true;
		depthStencilDesc.DepthWriteMask = zWrite == ZWrite::On ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
		depthStencilDesc.DepthFunc = static_cast<D3D11_COMPARISON_FUNC>(static_cast<uint32_t>(zTest) + 1);

		ComPtr<ID3D11DepthStencilState> state;
		HRESULT hr = m_Device->CreateDepthStencilState(&depthStencilDesc, state.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create depth stencil state."));
			return state.Get();
		}
		m_DepthStencilStates.push_back(std::make_pair(key, state));
		return state.Get();
	}

	D3D11_TEXTURE_ADDRESS_MODE GetAdressMode(const WrapMode& wrapMode)
	{
		if (wrapMode == WrapMode::Clamp)
		{
			return D3D11_TEXTURE_ADDRESS_CLAMP;
		}
		return D3D11_TEXTURE_ADDRESS_WRAP;
	}

	D3D11_FILTER GetFilter(const FilterMode& filterMode)
	{
		switch (filterMode)
		{
		case FilterMode::Point:	return D3D11_FILTER_MIN_MAG_MIP_POINT;
		case FilterMode::Bilinear: return D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
		case FilterMode::Trilinear: return D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		case FilterMode::Anisotropic: return D3D11_FILTER_ANISOTROPIC;
		case FilterMode::CompareDepth: return D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
		default: return D3D11_FILTER_MIN_MAG_MIP_POINT;
		}
	}

	D3D11_COMPARISON_FUNC GetComparison(const FilterMode& filterMode)
	{
		if (filterMode == FilterMode::CompareDepth)
		{
			return D3D11_COMPARISON_LESS;
		}
		return D3D11_COMPARISON_NEVER;
	}

	ID3D11SamplerState* GfxDeviceDX11::GetSamplerState(WrapMode wrapMode, FilterMode filterMode)
	{
		size_t key = static_cast<size_t>(wrapMode) << 8 | static_cast<size_t>(filterMode) << 16;
		for (auto& pair : m_SamplerStates)
		{
			if (pair.first == key)
			{
				return pair.second.Get();
			}
		}

		D3D11_TEXTURE_ADDRESS_MODE adress = GetAdressMode(wrapMode);
		D3D11_FILTER filter = GetFilter(filterMode);

		D3D11_SAMPLER_DESC samplerDesc = {};
		samplerDesc.Filter = filter;
		samplerDesc.AddressU = adress;
		samplerDesc.AddressV = adress;
		samplerDesc.AddressW = adress;
		samplerDesc.MipLODBias = 0.0f;
		samplerDesc.MaxAnisotropy = filter == D3D11_FILTER_ANISOTROPIC ? 8 : 1;
		samplerDesc.ComparisonFunc = GetComparison(filterMode);
		samplerDesc.MinLOD = -FLT_MAX;
		samplerDesc.MaxLOD = FLT_MAX;

		ComPtr<ID3D11SamplerState> state;
		HRESULT hr = m_Device->CreateSamplerState(&samplerDesc, state.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create sampler state."));
			return state.Get();
		}
		m_SamplerStates.push_back(std::make_pair(key, state));
		return state.Get();
	}
}