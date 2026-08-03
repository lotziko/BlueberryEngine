#include "GfxDeviceDX12.h"

#include "GfxShaderDX12.h"
#include "GfxComputeShaderDX12.h"
#include "GfxRayTracingShaderDX12.h"
#include "GfxBufferDX12.h"
#include "GfxTextureDX12.h"
#include "GfxBottomLevelAccelerationStructureDX12.h"
#include "GfxTopLevelAccelerationStructureDX12.h"
#include "GfxRayTracingShaderTableDX12.h"
#include "Blueberry\Graphics\Enums.h"
#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	#define DEBUG_LAYER true
	#define GPU_VALIDATION false

	bool GfxDeviceDX12::InitializeImpl(int width, int height, void* data)
	{
		if (!InitializeDirectX(*(static_cast<HWND*>(data)), width, height))
			return false;

		m_StateCache = GfxRenderStateCacheDX12(this);
		m_ComputeStateCache = GfxComputeRenderStateCacheDX12(this);
		m_RayTracingStateCache = GfxRayTracingRenderStateCacheDX12(this);

		return true;
	}

	void GfxDeviceDX12::ClearColorImpl(const Color& color)
	{
		if (m_BindedRenderTarget == nullptr)
		{
			BackbufferData& backbuffer = m_Backbuffers[m_BackbufferIndex];
			m_CommandList->ClearRenderTargetView(backbuffer.renderTargetView.GetCPU(), color, 0, nullptr);
		}
		else
		{
			m_BindedRenderTarget->SetState(D3D12_RESOURCE_STATE_RENDER_TARGET);
			m_CommandList->ClearRenderTargetView(m_BindedRenderTarget->GetRenderTargetView().GetCPU(), color, 0, nullptr);
		}
	}

	void GfxDeviceDX12::ClearDepthImpl(float depth)
	{
		if (m_BindedDepthStencil != nullptr)
		{
			m_BindedDepthStencil->SetState(D3D12_RESOURCE_STATE_DEPTH_WRITE);
			m_CommandList->ClearDepthStencilView(m_BindedDepthStencil->GetDepthStencilView().GetCPU(), D3D12_CLEAR_FLAG_DEPTH, depth, 0, 0, nullptr);
		}
	}

	//https://github.com/ocornut/imgui/blob/master/examples/example_win32_directx12/main.cpp
	void GfxDeviceDX12::WaitForFrameImpl()
	{
		WaitForSingleObjectEx(m_FrameLatencyWaitHandle, 16, TRUE);
		FrameContext& frameContext = m_FrameContexts[m_FrameIndex];
		frameContext.commandAllocator->Reset();
		m_CommandList->Reset(frameContext.commandAllocator.Get(), nullptr);
		ResizeBackbufferIfNeeded();
		Reset();
	}

	void GfxDeviceDX12::SwapBuffersImpl()
	{
		BackbufferData& backbuffer = m_Backbuffers[m_BackbufferIndex];
		if (backbuffer.state != D3D12_RESOURCE_STATE_PRESENT)
		{
			m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(backbuffer.resource.Get(), backbuffer.state, D3D12_RESOURCE_STATE_PRESENT));
			backbuffer.state = D3D12_RESOURCE_STATE_PRESENT;
		}
		m_CommandList->Close();

		ID3D12CommandList* lists[] = { m_CommandList.Get() };
		m_CommandQueue->ExecuteCommandLists(1, lists);
		m_SwapChain->Present(1, 0);

		UINT64 fenceValue = ++m_FenceLastSignaledValue;
		m_CommandQueue->Signal(m_Fence.Get(), fenceValue);
		m_BackbufferIndex = m_SwapChain->GetCurrentBackBufferIndex();

		UINT64 completedFenceValue = m_Fence->GetCompletedValue();
		if (completedFenceValue < m_FrameContexts[m_FrameIndex].fenceValue)
		{
			m_Fence->SetEventOnCompletion(m_FrameContexts[m_FrameIndex].fenceValue, m_FenceEvent);
			WaitForSingleObjectEx(m_FenceEvent, INFINITE, FALSE);
		}

		if (m_ReleasedResources.size() > 0)
		{
			for (auto it = m_ReleasedResources.begin(); it != m_ReleasedResources.end();)
			{
				if (m_FenceLastSignaledValue - it->first > 3)
				{
					it = m_ReleasedResources.erase(it);
				}
				else
				{
					++it;
				}
			}
		}
		m_FrameContexts[m_FrameIndex].fenceValue = fenceValue;
		m_FrameIndex = (m_FrameIndex + 1) % BUFFER_COUNT;
		m_UploadBuffer.UpdateGeneration(m_FenceLastSignaledValue);
	}

	void GfxDeviceDX12::SetViewportImpl(int x, int y, int width, int height)
	{
		D3D12_VIEWPORT viewport = {};
		viewport.TopLeftX = static_cast<FLOAT>(x);
		viewport.TopLeftY = static_cast<FLOAT>(y);
		viewport.Width = static_cast<FLOAT>(width);
		viewport.Height = static_cast<FLOAT>(height);
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;
		m_Viewport = viewport;

		m_CommandList->RSSetViewports(1, &viewport);

		D3D12_RECT rect = {};
		rect.left = x;
		rect.right = x + width;
		rect.top = y;
		rect.bottom = y + height;
		m_ScissorRect = rect;

		m_CommandList->RSSetScissorRects(1, &rect);
	}

	void GfxDeviceDX12::SetScissorRectImpl(int x, int y, int width, int height)
	{
		if (width > 0)
		{
			D3D12_RECT rect = {};
			rect.left = x;
			rect.right = x + width;
			rect.top = y;
			rect.bottom = y + height;
			m_ScissorRect = rect;

			m_CommandList->RSSetScissorRects(1, &rect);
		}
		else
		{
			m_ScissorRect = {};
			m_CommandList->RSSetScissorRects(0, NULL);
		}
	}

	void GfxDeviceDX12::ResizeBackbufferImpl(int width, int height)
	{
		m_BackbufferResizeRequest = Vector2Int(width, height);
	}

	uint32_t GfxDeviceDX12::GetViewCountImpl()
	{
		return m_ViewCount;
	}

	void GfxDeviceDX12::SetViewCountImpl(uint32_t count)
	{
		m_ViewCount = count;
	}

	void GfxDeviceDX12::SetDepthBiasImpl(uint32_t bias, float slopeBias)
	{
		m_DepthBias = bias;
		m_SlopeDepthBias = slopeBias;
	}

	bool GfxDeviceDX12::CreateVertexShaderImpl(const ByteData& vertexData, GfxVertexShader*& shader)
	{
		auto dxShader = new GfxVertexShaderDX12();
		if (!dxShader->Initialize(m_Device.Get(), vertexData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX12::CreateGeometryShaderImpl(const ByteData& geometryData, GfxGeometryShader*& shader)
	{
		auto dxShader = new GfxGeometryShaderDX12();
		if (!dxShader->Initialize(m_Device.Get(), geometryData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX12::CreateFragmentShaderImpl(const ByteData& fragmentData, GfxFragmentShader*& shader)
	{
		auto dxShader = new GfxFragmentShaderDX12();
		if (!dxShader->Initialize(m_Device.Get(), fragmentData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX12::CreateComputeShaderImpl(const ByteData& computeData, GfxComputeShader*& shader)
	{
		auto dxShader = new GfxComputeShaderDX12();
		if (!dxShader->Initialize(m_Device.Get(), computeData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX12::CreateRayTracingShaderImpl(const ByteData& rayTracingData, GfxRayTracingShader*& shader)
	{
		auto dxShader = new GfxRayTracingShaderDX12();
		if (!dxShader->Initialize(m_Device.Get(), rayTracingData))
		{
			return false;
		}
		shader = dxShader;
		return true;
	}

	bool GfxDeviceDX12::CreateBufferImpl(const BufferProperties& properties, GfxBuffer*& buffer)
	{
		GfxBufferDX12* dxBuffer = new GfxBufferDX12(this);
		if (!dxBuffer->Initialize(properties))
		{
			return false;
		}
		buffer = dxBuffer;
		return true;
	}

	bool GfxDeviceDX12::CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture)
	{
		GfxTextureDX12* dxTexture = new GfxTextureDX12(this);
		if (!dxTexture->Initialize(properties))
		{
			return false;
		}
		texture = dxTexture;
		return true;
	}

	bool GfxDeviceDX12::CreateBottomLevelAccelerationStructureImpl(const BottomLevelAccelerationStructureProperties& properties, GfxBottomLevelAccelerationStructure*& accelerationStructure)
	{
		GfxBottomLevelAccelerationStructureDX12* dxAccelerationStructure = new GfxBottomLevelAccelerationStructureDX12(this);
		if (!dxAccelerationStructure->Initialize(properties))
		{
			return false;
		}
		accelerationStructure = dxAccelerationStructure;
		return true;
	}

	bool GfxDeviceDX12::CreateTopLevelAccelerationStructureImpl(GfxTopLevelAccelerationStructure*& accelerationStructure)
	{
		GfxTopLevelAccelerationStructureDX12* dxAccelerationStructure = new GfxTopLevelAccelerationStructureDX12(this);
		accelerationStructure = dxAccelerationStructure;
		return true;
	}

	void GfxDeviceDX12::CopyImpl(GfxTexture* source, GfxTexture* target)
	{
		GfxTextureDX12* dxSource = static_cast<GfxTextureDX12*>(source);
		GfxTextureDX12* dxTarget = static_cast<GfxTextureDX12*>(target);
		dxSource->SetState(D3D12_RESOURCE_STATE_COPY_SOURCE);
		dxTarget->SetState(D3D12_RESOURCE_STATE_COPY_DEST);
		m_CommandList->CopyResource(dxTarget->GetResource(), dxSource->GetResource());
	}

	void GfxDeviceDX12::CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area)
	{
		GfxTextureDX12* dxSource = static_cast<GfxTextureDX12*>(source);
		GfxTextureDX12* dxTarget = static_cast<GfxTextureDX12*>(target);

		D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
		srcLocation.pResource = dxSource->GetResource();
		srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		srcLocation.SubresourceIndex = 0;

		D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
		dstLocation.pResource = dxTarget->GetResource();
		dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		dstLocation.SubresourceIndex = 0;

		D3D12_BOX srcBox = {};
		srcBox.left = static_cast<UINT>(area.x);
		srcBox.top = static_cast<UINT>(area.y);
		srcBox.right = static_cast<UINT>(area.x + area.width);
		srcBox.bottom = static_cast<UINT>(area.y + area.height);
		srcBox.front = 0;
		srcBox.back = 1;

		dxSource->SetState(D3D12_RESOURCE_STATE_COPY_SOURCE);
		dxTarget->SetState(D3D12_RESOURCE_STATE_COPY_DEST);
		m_CommandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, &srcBox);
	}

	void GfxDeviceDX12::CopyImpl(GfxTexture* source, GfxTexture* target, const Vector2Int& offset, const Rectangle& area)
	{
		GfxTextureDX12* dxSource = static_cast<GfxTextureDX12*>(source);
		GfxTextureDX12* dxTarget = static_cast<GfxTextureDX12*>(target);

		D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
		srcLocation.pResource = dxSource->GetResource();
		srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		srcLocation.SubresourceIndex = 0;

		D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
		dstLocation.pResource = dxTarget->GetResource();
		dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		dstLocation.SubresourceIndex = 0;

		D3D12_BOX srcBox = {};
		srcBox.left = static_cast<UINT>(area.x);
		srcBox.top = static_cast<UINT>(area.y);
		srcBox.right = static_cast<UINT>(area.x + area.width);
		srcBox.bottom = static_cast<UINT>(area.y + area.height);
		srcBox.front = 0;
		srcBox.back = 1;

		dxSource->SetState(D3D12_RESOURCE_STATE_COPY_SOURCE);
		dxTarget->SetState(D3D12_RESOURCE_STATE_COPY_DEST);
		m_CommandList->CopyTextureRegion(&dstLocation, static_cast<UINT>(offset.x), static_cast<UINT>(offset.y), 0, &srcLocation, &srcBox);
	}

	void GfxDeviceDX12::CopyImpl(GfxTexture* source, GfxTexture* target, uint32_t sourceSlice, uint32_t targetSlice, uint32_t mipLevel)
	{
		GfxTextureDX12* dxSource = static_cast<GfxTextureDX12*>(source);
		GfxTextureDX12* dxTarget = static_cast<GfxTextureDX12*>(target);

		UINT sourceSubresource = D3D12CalcSubresource(mipLevel, sourceSlice, 0, dxSource->GetMipLevels(), dxSource->GetArraySize());
		UINT targetSubresource = D3D12CalcSubresource(mipLevel, targetSlice, 0, dxTarget->GetMipLevels(), dxSource->GetArraySize());

		D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
		srcLocation.pResource = static_cast<GfxTextureDX12*>(source)->GetResource();
		srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		srcLocation.SubresourceIndex = sourceSubresource;

		D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
		dstLocation.pResource = static_cast<GfxTextureDX12*>(target)->GetResource();
		dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		dstLocation.SubresourceIndex = targetSubresource;

		dxSource->SetState(D3D12_RESOURCE_STATE_COPY_SOURCE);
		dxTarget->SetState(D3D12_RESOURCE_STATE_COPY_DEST);
		m_CommandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);
	}

	void GfxDeviceDX12::SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, uint32_t arraySlice, uint32_t mipLevel)
	{
		if (renderTexture == nullptr && depthStencilTexture == nullptr)
		{
			BackbufferData& backbuffer = m_Backbuffers[m_BackbufferIndex];
			if (backbuffer.state != D3D12_RESOURCE_STATE_RENDER_TARGET)
			{
				m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(backbuffer.resource.Get(), backbuffer.state, D3D12_RESOURCE_STATE_RENDER_TARGET));
				m_CommandList->OMSetRenderTargets(1, &backbuffer.renderTargetView.GetCPU(), FALSE, nullptr);
				backbuffer.state = D3D12_RESOURCE_STATE_RENDER_TARGET;
			}
			m_BindedRenderTarget = nullptr;
			m_BindedDepthStencil = nullptr;
			m_TargetInfo.renderTargetFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
			m_TargetInfo.depthStencilFormat = DXGI_FORMAT_UNKNOWN;
			m_TargetInfo.sampleCount = 1;
			m_TargetInfo.sampleQuality = 0;
		}
		else
		{
			if (m_BindedRenderTarget == nullptr && m_BindedDepthStencil == nullptr)
			{
				BackbufferData& backbuffer = m_Backbuffers[m_BackbufferIndex];
				if (backbuffer.state != D3D12_RESOURCE_STATE_PRESENT)
				{
					m_CommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(backbuffer.resource.Get(), backbuffer.state, D3D12_RESOURCE_STATE_PRESENT));
					backbuffer.state = D3D12_RESOURCE_STATE_PRESENT;
				}
			}

			D3D12_CPU_DESCRIPTOR_HANDLE renderTargets[1] = {};
			if (renderTexture != nullptr)
			{
				GfxTextureDX12* dxRenderTarget = static_cast<GfxTextureDX12*>(renderTexture);
				if (arraySlice || mipLevel)
				{
					renderTargets[0] = dxRenderTarget->GetRenderTargetView(arraySlice, mipLevel).GetCPU();
				}
				else
				{
					renderTargets[0] = dxRenderTarget->GetRenderTargetView().GetCPU();
				}

				m_BindedRenderTarget = dxRenderTarget;
				m_TargetInfo.renderTargetFormat = dxRenderTarget->GetDxgiFormat();
				m_TargetInfo.sampleCount = dxRenderTarget->GetAntiAliasing();
				m_TargetInfo.sampleQuality = dxRenderTarget->GetQuality();
			}
			else
			{
				m_TargetInfo.renderTargetFormat = DXGI_FORMAT_UNKNOWN;
			}
			D3D12_CPU_DESCRIPTOR_HANDLE depthStencil = {};
			if (depthStencilTexture != nullptr)
			{
				GfxTextureDX12* dxDepthStencil = static_cast<GfxTextureDX12*>(depthStencilTexture);
				depthStencil = dxDepthStencil->GetDepthStencilView().GetCPU();
				m_BindedDepthStencil = dxDepthStencil;
				m_TargetInfo.depthStencilFormat = dxDepthStencil->GetDxgiFormat();
				m_TargetInfo.sampleCount = dxDepthStencil->GetAntiAliasing();
				m_TargetInfo.sampleQuality = dxDepthStencil->GetQuality();
			}
			else
			{
				m_TargetInfo.depthStencilFormat = DXGI_FORMAT_UNKNOWN;
			}
			m_CommandList->OMSetRenderTargets(renderTexture == nullptr ? 0 : 1, renderTexture == nullptr ? nullptr : renderTargets, FALSE, depthStencilTexture == nullptr ? nullptr : &depthStencil);
		}
	}

	void GfxDeviceDX12::SetGlobalBufferImpl(size_t id, GfxBuffer* buffer)
	{
		auto dxBuffer = static_cast<GfxBufferDX12*>(buffer);
		for (auto& pair : m_BindedBuffers)
		{
			if (pair.first == id)
			{
				pair.second = dxBuffer->GetIndex();
				return;
			}
		}
		m_BindedBuffers.push_back(std::make_pair(id, dxBuffer->GetIndex()));
	}

	void GfxDeviceDX12::SetGlobalTextureImpl(size_t id, GfxTexture* texture)
	{
		auto dxTexture = static_cast<GfxTextureDX12*>(texture);
		for (auto& pair : m_BindedTextures)
		{
			if (pair.first == id)
			{
				pair.second = dxTexture->GetIndex();
				return;
			}
		}
		m_BindedTextures.push_back(std::make_pair(id, dxTexture->GetIndex()));
	}

	D3D12_PRIMITIVE_TOPOLOGY GetPrimitiveTopologyD3D12(const Topology& topology)
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

	void GfxDeviceDX12::DrawImpl(const GfxDrawingOperation& operation)
	{
		if (!operation.IsValid())
		{
			return;
		}

		const GfxRenderStateDX12 renderState = m_StateCache.GetRenderState(operation.material, operation.passId, operation.layout, m_TargetInfo, operation.topology, m_DepthBias, m_SlopeDepthBias, operation.isCounterClockwise, operation.isSolid);

		if (!renderState.isValid)
		{
			return;
		}

		if (m_BindedRenderTarget != nullptr)
		{
			m_BindedRenderTarget->SetState(D3D12_RESOURCE_STATE_RENDER_TARGET);
		}

		if (m_BindedDepthStencil != nullptr)
		{
			m_BindedDepthStencil->SetState(D3D12_RESOURCE_STATE_DEPTH_WRITE);
		}

		if (renderState.pipelineState != m_PipelineState)
		{
			m_PipelineState = renderState.pipelineState;
			m_CommandList->SetPipelineState(renderState.pipelineState);
		}
		
		if (renderState.vertexConstantBuffersCount > 0)
		{
			if (memcmp(m_RenderState.vertexConstantBuffers, renderState.vertexConstantBuffers, sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * renderState.vertexConstantBuffersCount) != 0)
			{
				auto vertexÑbvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.vertexConstantBuffersCount);
				m_Device->CopyDescriptors(1, &vertexÑbvDestHandle.GetCPU(), &renderState.vertexConstantBuffersCount, renderState.vertexConstantBuffersCount, renderState.vertexConstantBuffers, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
				m_CommandList->SetGraphicsRootDescriptorTable(0, vertexÑbvDestHandle.GetGPU());
			}
		}

		if (renderState.geometryConstantBuffersCount > 0)
		{
			if (memcmp(m_RenderState.geometryConstantBuffers, renderState.geometryConstantBuffers, sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * renderState.geometryConstantBuffersCount) != 0)
			{
				auto geometryÑbvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.geometryConstantBuffersCount);
				m_Device->CopyDescriptors(1, &geometryÑbvDestHandle.GetCPU(), &renderState.geometryConstantBuffersCount, renderState.geometryConstantBuffersCount, renderState.geometryConstantBuffers, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
				m_CommandList->SetGraphicsRootDescriptorTable(1, geometryÑbvDestHandle.GetGPU());
			}
		}

		if (renderState.pixelConstantBuffersCount > 0)
		{
			if (memcmp(m_RenderState.pixelConstantBuffers, renderState.pixelConstantBuffers, sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * renderState.pixelConstantBuffersCount) != 0)
			{
				auto pixelÑbvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.pixelConstantBuffersCount);
				m_Device->CopyDescriptors(1, &pixelÑbvDestHandle.GetCPU(), &renderState.pixelConstantBuffersCount, renderState.pixelConstantBuffersCount, renderState.pixelConstantBuffers, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
				m_CommandList->SetGraphicsRootDescriptorTable(2, pixelÑbvDestHandle.GetGPU());
			}
		}

		if (renderState.vertexShaderResourceViewsCount > 0)
		{
			if (memcmp(m_RenderState.vertexShaderResourceViews, renderState.vertexShaderResourceViews, sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * renderState.vertexShaderResourceViewsCount) != 0)
			{
				auto vertexSrvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.vertexShaderResourceViewsCount);
				m_Device->CopyDescriptors(1, &vertexSrvDestHandle.GetCPU(), &renderState.vertexShaderResourceViewsCount, renderState.vertexShaderResourceViewsCount, renderState.vertexShaderResourceViews, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
				m_CommandList->SetGraphicsRootDescriptorTable(3, vertexSrvDestHandle.GetGPU());
			}
		}

		if (renderState.pixelShaderResourceViewsCount > 0)
		{
			if (memcmp(m_RenderState.pixelShaderResourceViews, renderState.pixelShaderResourceViews, sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) * renderState.pixelShaderResourceViewsCount) != 0)
			{
				auto pixelSrvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.pixelShaderResourceViewsCount);
				m_Device->CopyDescriptors(1, &pixelSrvDestHandle.GetCPU(), &renderState.pixelShaderResourceViewsCount, renderState.pixelShaderResourceViewsCount, renderState.pixelShaderResourceViews, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
				m_CommandList->SetGraphicsRootDescriptorTable(4, pixelSrvDestHandle.GetGPU());
			}
		}

		if (renderState.vertexSamplersCount > 0)
		{
			uint32_t offset = GetSamplersOffset(renderState.vertexSamplers, renderState.vertexSamplersCount);
			m_CommandList->SetGraphicsRootDescriptorTable(5, m_SamplerRingHeap.GetGPU(offset));
		}

		if (renderState.pixelSamplersCount > 0)
		{
			uint32_t offset = GetSamplersOffset(renderState.pixelSamplers, renderState.pixelSamplersCount);
			m_CommandList->SetGraphicsRootDescriptorTable(6, m_SamplerRingHeap.GetGPU(offset));
		}

		if (renderState.pixelBindlessIndexesCount > 0)
		{
			m_CommandList->SetGraphicsRoot32BitConstants(7, renderState.pixelBindlessIndexesCount, renderState.bindlessIndexes, 0);
		}
		
		if (operation.topology != m_Topology)
		{
			m_Topology = operation.topology;
			m_CommandList->IASetPrimitiveTopology(GetPrimitiveTopologyD3D12(operation.topology));
		}

		auto dxVertexBuffer = static_cast<GfxBufferDX12*>(operation.vertexBuffer);
		dxVertexBuffer->SetState(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		if (dxVertexBuffer != m_VertexBuffer)
		{
			m_VertexBuffer = dxVertexBuffer;
			m_CommandList->IASetVertexBuffers(0, 1, &dxVertexBuffer->GetVertexView());
		}

		auto dxInstanceBuffer = static_cast<GfxBufferDX12*>(operation.instanceBuffer);
		if (dxInstanceBuffer != m_InstanceBuffer || operation.instanceOffset != m_InstanceOffset)
		{
			m_InstanceBuffer = dxInstanceBuffer;
			m_InstanceOffset = operation.instanceOffset;
			if (dxInstanceBuffer != nullptr)
			{
				dxInstanceBuffer->SetState(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
				uint32_t byteOffset = m_InstanceBuffer ? m_InstanceOffset * m_InstanceBuffer->GetElementSize() : 0;
				D3D12_VERTEX_BUFFER_VIEW view = dxInstanceBuffer->GetVertexView();
				view.BufferLocation += static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(byteOffset);
				view.SizeInBytes -= byteOffset;
				m_CommandList->IASetVertexBuffers(1, 1, &view);
			}
		}

		if (operation.indexBuffer == nullptr)
		{
			if (m_InstanceBuffer == nullptr)
			{
				m_CommandList->DrawInstanced(operation.vertexCount, m_ViewCount, 0, 0);
			}
			else
			{
				m_CommandList->DrawInstanced(operation.vertexCount, operation.instanceCount * m_ViewCount, 0, 0);
			}
		}
		else
		{
			auto dxIndexBuffer = static_cast<GfxBufferDX12*>(operation.indexBuffer);
			dxIndexBuffer->SetState(D3D12_RESOURCE_STATE_INDEX_BUFFER);
			if (dxIndexBuffer != m_IndexBuffer)
			{
				m_IndexBuffer = dxIndexBuffer;
				m_CommandList->IASetIndexBuffer(&dxIndexBuffer->GetIndexView());
			}
			if (m_InstanceBuffer == nullptr)
			{
				m_CommandList->DrawIndexedInstanced(operation.indexCount, m_ViewCount, operation.indexOffset, 0, 0);
			}
			else
			{
				m_CommandList->DrawIndexedInstanced(operation.indexCount, operation.instanceCount * m_ViewCount, operation.indexOffset, 0, 0);
			}
		}
		m_RenderState = renderState;
	}

	void GfxDeviceDX12::DispatchImpl(ComputeShader* shader, uint32_t kernelIndex, uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ)
	{
		const GfxComputeRenderStateDX12 renderState = m_ComputeStateCache.GetRenderState(shader, kernelIndex);

		if (!renderState.isValid)
		{
			return;
		}
		
		if (renderState.pipelineState != m_PipelineState)
		{
			m_PipelineState = renderState.pipelineState;
			m_CommandList->SetPipelineState(renderState.pipelineState);
		}

		if (renderState.constantBuffersCount > 0)
		{
			auto cbvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.constantBuffersCount);
			m_Device->CopyDescriptors(1, &cbvDestHandle.GetCPU(), &renderState.constantBuffersCount, renderState.constantBuffersCount, renderState.constantBuffers, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			m_CommandList->SetComputeRootDescriptorTable(0, cbvDestHandle.GetGPU());
		}

		if (renderState.shaderResourceViewsCount > 0)
		{
			auto srvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.shaderResourceViewsCount);
			m_Device->CopyDescriptors(1, &srvDestHandle.GetCPU(), &renderState.shaderResourceViewsCount, renderState.shaderResourceViewsCount, renderState.shaderResourceViews, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			m_CommandList->SetComputeRootDescriptorTable(1, srvDestHandle.GetGPU());
		}

		if (renderState.unorderedAccessViewsCount > 0)
		{
			auto uavDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.unorderedAccessViewsCount);
			m_Device->CopyDescriptors(1, &uavDestHandle.GetCPU(), &renderState.unorderedAccessViewsCount, renderState.unorderedAccessViewsCount, renderState.unorderedAccessViews, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			m_CommandList->SetComputeRootDescriptorTable(2, uavDestHandle.GetGPU());
		}

		if (renderState.samplersCount > 0)
		{
			uint32_t offset = GetSamplersOffset(renderState.samplers, renderState.samplersCount);
			m_CommandList->SetComputeRootDescriptorTable(3, m_SamplerRingHeap.GetGPU(offset));
		}
		
		m_CommandList->Dispatch(threadGroupsX, threadGroupsY, threadGroupsZ);
	}

	void GfxDeviceDX12::DispatchRaysImpl(RayTracingShader* shader, GfxTopLevelAccelerationStructure* accelerationStructure, uint32_t width, uint32_t height, uint32_t depth)
	{
		const GfxRayTracingRenderStateDX12 renderState = m_RayTracingStateCache.GetRenderState(shader, accelerationStructure);

		auto dxAccelerationStructure = static_cast<GfxTopLevelAccelerationStructureDX12*>(accelerationStructure);
		
		D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
		dispatchDesc.RayGenerationShaderRecord.StartAddress = renderState.rayGenerationShaderTableAddress;
		dispatchDesc.RayGenerationShaderRecord.SizeInBytes = renderState.rayGenerationShaderTableSize;
		dispatchDesc.HitGroupTable.StartAddress = renderState.hitGroupShaderTableAddress;
		dispatchDesc.HitGroupTable.SizeInBytes = renderState.hitGroupShaderTableSize;
		dispatchDesc.HitGroupTable.StrideInBytes = renderState.hitGroupShaderTableStride;
		dispatchDesc.MissShaderTable.StartAddress = renderState.missShaderTableAddress;
		dispatchDesc.MissShaderTable.SizeInBytes = renderState.missShaderTableSize;
		dispatchDesc.MissShaderTable.StrideInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
		dispatchDesc.Width = width;
		dispatchDesc.Height = height;
		dispatchDesc.Depth = depth;

		m_DxrCommandList->SetComputeRootSignature(m_DxrGlobalRootSignature.Get());
		
		if (renderState.constantBuffersCount > 0)
		{
			auto cbvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.constantBuffersCount);
			m_Device->CopyDescriptors(1, &cbvDestHandle.GetCPU(), &renderState.constantBuffersCount, renderState.constantBuffersCount, renderState.constantBuffers, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			m_CommandList->SetComputeRootDescriptorTable(0, cbvDestHandle.GetGPU());
		}

		if (renderState.shaderResourceViewsCount > 0)
		{
			auto srvDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.shaderResourceViewsCount);
			m_Device->CopyDescriptors(1, &srvDestHandle.GetCPU(), &renderState.shaderResourceViewsCount, renderState.shaderResourceViewsCount, renderState.shaderResourceViews, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			m_CommandList->SetComputeRootDescriptorTable(1, srvDestHandle.GetGPU());
		}

		if (renderState.unorderedAccessViewsCount > 0)
		{
			auto uavDestHandle = m_CbvSrvUavRingHeap.AllocateTemporary(renderState.unorderedAccessViewsCount);
			m_Device->CopyDescriptors(1, &uavDestHandle.GetCPU(), &renderState.unorderedAccessViewsCount, renderState.unorderedAccessViewsCount, renderState.unorderedAccessViews, nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			m_CommandList->SetComputeRootDescriptorTable(2, uavDestHandle.GetGPU());
		}

		if (renderState.samplers > 0)
		{
			uint32_t offset = GetSamplersOffset(renderState.samplers, renderState.samplersCount);
			m_CommandList->SetComputeRootDescriptorTable(3, m_SamplerRingHeap.GetGPU(offset));
		}

		m_DxrCommandList->SetPipelineState1(renderState.stateObject);
		m_DxrCommandList->DispatchRays(&dispatchDesc);
		m_DxrCommandList->SetComputeRootSignature(m_ComputeRootSignature.Get());
	}

	Matrix GfxDeviceDX12::GetGPUMatrixImpl(const Matrix& matrix) const
	{
		return matrix;
	}

	ID3D12Device* GfxDeviceDX12::GetDevice()
	{
		return m_Device.Get();
	}

	ID3D12CommandQueue* GfxDeviceDX12::GetCommandQueue()
	{
		return m_CommandQueue.Get();
	}

	ID3D12GraphicsCommandList* GfxDeviceDX12::GetCommandList()
	{
		return m_CommandList.Get();
	}

	ID3D12RootSignature* GfxDeviceDX12::GetGraphicsRootSignature()
	{
		return m_GraphicsRootSignature.Get();
	}

	ID3D12RootSignature* GfxDeviceDX12::GetComputeRootSignature()
	{
		return m_ComputeRootSignature.Get();
	}

	ID3D12Device5* GfxDeviceDX12::GetDxrDevice()
	{
		return m_DxrDevice.Get();
	}

	ID3D12GraphicsCommandList4* GfxDeviceDX12::GetDxrCommandList()
	{
		return m_DxrCommandList.Get();
	}

	ID3D12RootSignature* GfxDeviceDX12::GetDxrGlobalRootSignature()
	{
		return m_DxrGlobalRootSignature.Get();
	}

	ID3D12RootSignature* GfxDeviceDX12::GetDxrLocalRootSignature()
	{
		return m_DxrLocalRootSignature.Get();
	}

	HWND GfxDeviceDX12::GetHwnd()
	{
		return m_Hwnd;
	}

	GfxDescriptorHeapDX12& GfxDeviceDX12::GetCbvSrvUavHeap()
	{
		return m_CbvSrvUavHeap;
	}

	GfxDescriptorHeapDX12& GfxDeviceDX12::GetRtvHeap()
	{
		return m_RtvHeap;
	}

	GfxDescriptorHeapDX12& GfxDeviceDX12::GetDsvHeap()
	{
		return m_DsvHeap;
	}

	GfxDescriptorHeapDX12& GfxDeviceDX12::GetCbvSrvUavRingHeap()
	{
		return m_CbvSrvUavRingHeap;
	}

	GfxUploadBufferDX12& GfxDeviceDX12::GetUploadBuffer()
	{
		return m_UploadBuffer;
	}

	GfxReadbackBufferDX12& GfxDeviceDX12::GetReadbackBuffer()
	{
		return m_ReadbackBuffer;
	}

	uint64_t GfxDeviceDX12::GetGeneration()
	{
		return m_FenceLastSignaledValue;
	}

	void GfxDeviceDX12::WaitForGPU()
	{
		m_CommandList->Close();
		ID3D12CommandList* lists[] = { m_CommandList.Get() };
		m_CommandQueue->ExecuteCommandLists(1, lists);
		FrameContext& frameContext = m_FrameContexts[m_FrameIndex];
		UINT64 fenceValue = ++m_FenceLastSignaledValue;
		m_CommandQueue->Signal(m_Fence.Get(), fenceValue);
		m_Fence->SetEventOnCompletion(fenceValue, m_FenceEvent);
		WaitForSingleObject(m_FenceEvent, INFINITE);
		m_FrameContexts[m_FrameIndex].fenceValue = fenceValue;
		frameContext.commandAllocator->Reset();
		m_CommandList->Reset(frameContext.commandAllocator.Get(), nullptr);
	}

	void GfxDeviceDX12::Reset()
	{
		ID3D12DescriptorHeap* heaps[] = { m_CbvSrvUavRingHeap.GetHeap(), m_SamplerRingHeap.GetHeap() };
		m_CommandList->SetDescriptorHeaps(2, heaps);
		m_CommandList->SetGraphicsRootSignature(m_GraphicsRootSignature.Get());
		m_CommandList->SetComputeRootSignature(m_ComputeRootSignature.Get());
		BackbufferData& backbuffer = m_Backbuffers[m_BackbufferIndex];
		m_CommandList->OMSetRenderTargets((m_BindedRenderTarget == nullptr && m_BindedDepthStencil != nullptr) ? 0 : 1, m_BindedRenderTarget == nullptr ? &backbuffer.renderTargetView.GetCPU() : &m_BindedRenderTarget->GetRenderTargetView().GetCPU(), FALSE, m_BindedDepthStencil == nullptr ? nullptr : &m_BindedDepthStencil->GetDepthStencilView().GetCPU());
		m_CommandList->RSSetViewports(1, &m_Viewport);
		if (m_ScissorRect.right > 0)
		{
			m_CommandList->RSSetScissorRects(1, &m_ScissorRect);
		}
		m_PipelineState = nullptr;
		m_VertexBuffer = nullptr;
		m_IndexBuffer = nullptr;
		m_IndexBuffer = nullptr;
		m_Topology = (Topology)-1;
		m_RenderState = {};
	}

	void GfxDeviceDX12::Release(ComPtr<ID3D12Resource>& resource)
	{
		if (resource.Get() != nullptr)
		{
			m_ReleasedResources.push_back(std::make_pair(m_FenceLastSignaledValue, resource));
		}
	}

	bool GfxDeviceDX12::InitializeDirectX(HWND hwnd, int width, int height)
	{
		m_Hwnd = hwnd;

		UINT dxgiFactoryFlags = 0;

		ComPtr<IDXGIFactory6> dxgiFactory;
		HRESULT hr = CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&dxgiFactory));

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

#if DEBUG_LAYER
		ID3D12Debug1* debug = nullptr;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
		{
			debug->EnableDebugLayer();
#if GPU_VALIDATION
			debug->SetEnableGPUBasedValidation(true);
#endif
		}
#endif

		hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_Device));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating device."));
			return false;
		}

		m_Device->QueryInterface(IID_PPV_ARGS(&m_DxrDevice));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error getting DXR device."));
			return false;
		}

#if DEBUG_LAYER
		if (debug != nullptr)
		{
			ID3D12InfoQueue* infoQueue = nullptr;
			m_Device->QueryInterface(IID_PPV_ARGS(&infoQueue));
			infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
			infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);

			const int D3D12_MESSAGE_ID_FENCE_ZERO_WAIT_ = 1424; 
			D3D12_MESSAGE_ID disabledMessages[] = { (D3D12_MESSAGE_ID)D3D12_MESSAGE_ID_FENCE_ZERO_WAIT_ };
			D3D12_INFO_QUEUE_FILTER filter = {};
			filter.DenyList.NumIDs = 1;
			filter.DenyList.pIDList = disabledMessages;
			infoQueue->AddStorageFilterEntries(&filter);

			infoQueue->Release();
			debug->Release();
		}
#endif

		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

		hr = m_Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_CommandQueue));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating command queue."));
			return false;
		}

		DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
		swapChainDesc.BufferCount = BUFFER_COUNT;
		swapChainDesc.Width = width;
		swapChainDesc.Height = height;
		swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapChainDesc.SampleDesc.Count = 1;

		ComPtr<IDXGISwapChain1> swapChain;
		hr = dxgiFactory->CreateSwapChainForHwnd(
			m_CommandQueue.Get(),
			hwnd,
			&swapChainDesc,
			nullptr,
			nullptr,
			&swapChain
		);

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating swapchain."));
			return false;
		}

		hr = swapChain.As(&m_SwapChain);

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error getting swapchain3."));
			return false;
		}

		m_CbvSrvUavHeap = GfxDescriptorHeapDX12(this);
		if (!m_CbvSrvUavHeap.Initialize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, false, 1024 * 16, 0))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating cbv srv uav heap."));
			return false;
		}

		m_RtvHeap = GfxDescriptorHeapDX12(this);
		if (!m_RtvHeap.Initialize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false, 1024, 0))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating rtv heap."));
			return false;
		}

		m_DsvHeap = GfxDescriptorHeapDX12(this);
		if (!m_DsvHeap.Initialize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false, 64, 0))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating dsv heap."));
			return false;
		}

		m_SamplerHeap = GfxDescriptorHeapDX12(this);
		if (!m_SamplerHeap.Initialize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, false, 128, 0))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating sampler heap."));
			return false;
		}

		m_CbvSrvUavRingHeap = GfxDescriptorHeapDX12(this);
		if (!m_CbvSrvUavRingHeap.Initialize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, 1024 * 16, 1024 * 64))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating cbv srv uav ring heap."));
			return false;
		}

		m_SamplerRingHeap = GfxDescriptorHeapDX12(this);
		if (!m_SamplerRingHeap.Initialize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true, 128, 1920))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating sampler ring heap."));
			return false;
		}

		for (UINT i = 0; i < BUFFER_COUNT; i++)
		{
			BackbufferData& backbuffer = m_Backbuffers[i];
			hr = m_SwapChain->GetBuffer(i, IID_PPV_ARGS(&backbuffer.resource));
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error getting swapchain buffer."));
				return false;
			}
			GfxHandleDX12 handle = m_RtvHeap.AllocatePersistent();
			m_Device->CreateRenderTargetView(backbuffer.resource.Get(), nullptr, handle.GetCPU());
			backbuffer.renderTargetView = handle;
			backbuffer.state = D3D12_RESOURCE_STATE_PRESENT;
		}

		for (UINT i = 0; i < BUFFER_COUNT; i++)
		{
			hr = m_Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_FrameContexts[i].commandAllocator));

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating command allocator."));
				return false;
			}
		}

		hr = m_Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_FrameContexts[0].commandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_CommandList));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating command list."));
			return false;
		}

		m_CommandList->QueryInterface(IID_PPV_ARGS(&m_DxrCommandList));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error getting DXR command list."));
			return false;
		}

		hr = m_CommandList->Close();

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error closing command list."));
			return false;
		}

		m_UploadBuffer = GfxUploadBufferDX12(this);
		m_ReadbackBuffer = GfxReadbackBufferDX12(this);

		hr = m_Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_Fence));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating fence."));
			return false;
		}

		m_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

		if (m_FenceEvent == nullptr)
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating fence event."));
			return false;
		}

		// Graphics root signature
		{
			D3D12_DESCRIPTOR_RANGE cbvRange = {};
			cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
			cbvRange.NumDescriptors = 8;
			cbvRange.BaseShaderRegister = 0;
			cbvRange.RegisterSpace = 0;
			cbvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_DESCRIPTOR_RANGE srvRange = {};
			srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
			srvRange.NumDescriptors = 16;
			srvRange.BaseShaderRegister = 0;
			srvRange.RegisterSpace = 0;
			srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_DESCRIPTOR_RANGE samplerRange = {};
			samplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
			samplerRange.NumDescriptors = 16;
			samplerRange.BaseShaderRegister = 0;
			samplerRange.RegisterSpace = 0;
			samplerRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_ROOT_PARAMETER vertexCbvParam = {};
			vertexCbvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			vertexCbvParam.DescriptorTable.NumDescriptorRanges = 1;
			vertexCbvParam.DescriptorTable.pDescriptorRanges = &cbvRange;
			vertexCbvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

			D3D12_ROOT_PARAMETER geometryCbvParam = {};
			geometryCbvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			geometryCbvParam.DescriptorTable.NumDescriptorRanges = 1;
			geometryCbvParam.DescriptorTable.pDescriptorRanges = &cbvRange;
			geometryCbvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_GEOMETRY;

			D3D12_ROOT_PARAMETER pixelCbvParam = {};
			pixelCbvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			pixelCbvParam.DescriptorTable.NumDescriptorRanges = 1;
			pixelCbvParam.DescriptorTable.pDescriptorRanges = &cbvRange;
			pixelCbvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

			D3D12_ROOT_PARAMETER vertexSrvParam = {};
			vertexSrvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			vertexSrvParam.DescriptorTable.NumDescriptorRanges = 1;
			vertexSrvParam.DescriptorTable.pDescriptorRanges = &srvRange;
			vertexSrvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

			D3D12_ROOT_PARAMETER pixelSrvParam = {};
			pixelSrvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			pixelSrvParam.DescriptorTable.NumDescriptorRanges = 1;
			pixelSrvParam.DescriptorTable.pDescriptorRanges = &srvRange;
			pixelSrvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

			D3D12_ROOT_PARAMETER vertexSamplerParam = {};
			vertexSamplerParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			vertexSamplerParam.DescriptorTable.NumDescriptorRanges = 1;
			vertexSamplerParam.DescriptorTable.pDescriptorRanges = &samplerRange;
			vertexSamplerParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

			D3D12_ROOT_PARAMETER pixelSamplerParam = {};
			pixelSamplerParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			pixelSamplerParam.DescriptorTable.NumDescriptorRanges = 1;
			pixelSamplerParam.DescriptorTable.pDescriptorRanges = &samplerRange;
			pixelSamplerParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

			D3D12_ROOT_PARAMETER materialDataCBVParam = {};
			materialDataCBVParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
			materialDataCBVParam.Constants.ShaderRegister = 0;
			materialDataCBVParam.Constants.RegisterSpace = 1;
			materialDataCBVParam.Constants.Num32BitValues = 16;
			materialDataCBVParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

			D3D12_ROOT_PARAMETER params[] =
			{
				vertexCbvParam,
				geometryCbvParam,
				pixelCbvParam,
				vertexSrvParam,
				pixelSrvParam,
				vertexSamplerParam,
				pixelSamplerParam,
				materialDataCBVParam
			};

			D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
			rootSignatureDesc.NumParameters = _countof(params);
			rootSignatureDesc.pParameters = params;
			rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;
			hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &signature, &error);
			
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error serializing graphics root signature.") << '\n' << static_cast<const char*>(error->GetBufferPointer()));
				return false;
			}
			
			hr = m_Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_GraphicsRootSignature));
		
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating graphics root signature."));
				return false;
			}
		}

		// Compute root signature
		{
			D3D12_DESCRIPTOR_RANGE cbvRange = {};
			cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
			cbvRange.NumDescriptors = 14;
			cbvRange.BaseShaderRegister = 0;
			cbvRange.RegisterSpace = 0;
			cbvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_DESCRIPTOR_RANGE srvRange = {};
			srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
			srvRange.NumDescriptors = 16;
			srvRange.BaseShaderRegister = 0;
			srvRange.RegisterSpace = 0;
			srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_DESCRIPTOR_RANGE uavRange = {};
			uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
			uavRange.NumDescriptors = 8;
			uavRange.BaseShaderRegister = 0;
			uavRange.RegisterSpace = 0;
			uavRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_DESCRIPTOR_RANGE samplerRange = {};
			samplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
			samplerRange.NumDescriptors = 16;
			samplerRange.BaseShaderRegister = 0;
			samplerRange.RegisterSpace = 0;
			samplerRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_ROOT_PARAMETER cbvParam = {};
			cbvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			cbvParam.DescriptorTable.NumDescriptorRanges = 1;
			cbvParam.DescriptorTable.pDescriptorRanges = &cbvRange;
			cbvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER srvParam = {};
			srvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			srvParam.DescriptorTable.NumDescriptorRanges = 1;
			srvParam.DescriptorTable.pDescriptorRanges = &srvRange;
			srvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER uavParam = {};
			uavParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			uavParam.DescriptorTable.NumDescriptorRanges = 1;
			uavParam.DescriptorTable.pDescriptorRanges = &uavRange;
			uavParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER samplerParam = {};
			samplerParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			samplerParam.DescriptorTable.NumDescriptorRanges = 1;
			samplerParam.DescriptorTable.pDescriptorRanges = &samplerRange;
			samplerParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER params[] =
			{
				cbvParam,
				srvParam,
				uavParam,
				samplerParam
			};

			D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
			rootSignatureDesc.NumParameters = _countof(params);
			rootSignatureDesc.pParameters = params;
			rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;
			hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &signature, &error);

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error serializing compute root signature.") << '\n' << static_cast<const char*>(error->GetBufferPointer()));
				return false;
			}

			hr = m_Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_ComputeRootSignature));

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating compute root signature."));
				return false;
			}
		}

		// DXR global root signature
		{
			D3D12_DESCRIPTOR_RANGE cbvRange = {};
			cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
			cbvRange.NumDescriptors = 4;
			cbvRange.BaseShaderRegister = 0;
			cbvRange.RegisterSpace = 0;
			cbvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_DESCRIPTOR_RANGE srvRange = {};
			srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
			srvRange.NumDescriptors = 4;
			srvRange.BaseShaderRegister = 0;
			srvRange.RegisterSpace = 0;
			srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_DESCRIPTOR_RANGE uavRange = {};
			uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
			uavRange.NumDescriptors = 4;
			uavRange.BaseShaderRegister = 0;
			uavRange.RegisterSpace = 0;
			uavRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_DESCRIPTOR_RANGE samplerRange = {};
			samplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
			samplerRange.NumDescriptors = 4;
			samplerRange.BaseShaderRegister = 0;
			samplerRange.RegisterSpace = 0;
			samplerRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

			D3D12_ROOT_PARAMETER cbvParam = {};
			cbvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			cbvParam.DescriptorTable.NumDescriptorRanges = 1;
			cbvParam.DescriptorTable.pDescriptorRanges = &cbvRange;
			cbvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER srvParam = {};
			srvParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			srvParam.DescriptorTable.NumDescriptorRanges = 1;
			srvParam.DescriptorTable.pDescriptorRanges = &srvRange;
			srvParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER uavParam = {};
			uavParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			uavParam.DescriptorTable.NumDescriptorRanges = 1;
			uavParam.DescriptorTable.pDescriptorRanges = &uavRange;
			uavParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER samplerParam = {};
			samplerParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			samplerParam.DescriptorTable.NumDescriptorRanges = 1;
			samplerParam.DescriptorTable.pDescriptorRanges = &samplerRange;
			samplerParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER params[] =
			{
				cbvParam,
				srvParam,
				uavParam,
				samplerParam
			};

			D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
			rootSignatureDesc.NumParameters = _countof(params);
			rootSignatureDesc.pParameters = params;
			rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;
			hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &signature, &error);

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error serializing dxr global root signature.") << '\n' << static_cast<const char*>(error->GetBufferPointer()));
				return false;
			}

			hr = m_Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_DxrGlobalRootSignature));

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating dxr global root signature."));
				return false;
			}
		}

		// DXR local root signature
		{
			D3D12_ROOT_PARAMETER vertexBufferParam = {};
			vertexBufferParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
			vertexBufferParam.Descriptor.ShaderRegister = 0;
			vertexBufferParam.Descriptor.RegisterSpace = 1;
			vertexBufferParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER indexBufferParam = {};
			indexBufferParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
			indexBufferParam.Descriptor.ShaderRegister = 1;
			indexBufferParam.Descriptor.RegisterSpace = 1;
			indexBufferParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER meshDataCBVParam = {};
			meshDataCBVParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
			meshDataCBVParam.Constants.ShaderRegister = 0;
			meshDataCBVParam.Constants.RegisterSpace = 1;
			meshDataCBVParam.Constants.Num32BitValues = 3;
			meshDataCBVParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER materialDataCBVParam = {};
			materialDataCBVParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
			materialDataCBVParam.Constants.ShaderRegister = 1;
			materialDataCBVParam.Constants.RegisterSpace = 1;
			materialDataCBVParam.Constants.Num32BitValues = 16;
			materialDataCBVParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

			D3D12_ROOT_PARAMETER params[] =
			{
				vertexBufferParam,
				indexBufferParam,
				meshDataCBVParam,
				materialDataCBVParam
			};

			D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
			rootSignatureDesc.NumParameters = _countof(params);
			rootSignatureDesc.pParameters = params;
			rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;
			hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &signature, &error);

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error serializing dxr local root signature.") << '\n' << static_cast<const char*>(error->GetBufferPointer()));
				return false;
			}

			hr = m_Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_DxrLocalRootSignature));

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error creating dxr local root signature."));
				return false;
			}
		}

		m_BindedRenderTarget = nullptr;
		m_BindedDepthStencil = nullptr;

		BB_INFO("DirectX initialized successful.");

		return true;
	}

	D3D12_TEXTURE_ADDRESS_MODE GetAdressModeD3D12(const WrapMode& wrapMode)
	{
		if (wrapMode == WrapMode::Clamp)
		{
			return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		}
		return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	}

	D3D12_FILTER GetFilterD3D12(const FilterMode& filterMode)
	{
		switch (filterMode)
		{
		case FilterMode::Point:	return D3D12_FILTER_MIN_MAG_MIP_POINT;
		case FilterMode::Bilinear: return D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
		case FilterMode::Trilinear: return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
		case FilterMode::Anisotropic: return D3D12_FILTER_ANISOTROPIC;
		case FilterMode::CompareDepth: return D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
		default: return D3D12_FILTER_MIN_MAG_MIP_POINT;
		}
	}

	D3D12_COMPARISON_FUNC GetComparisonD3D12(const FilterMode& filterMode)
	{
		if (filterMode == FilterMode::CompareDepth)
		{
			return D3D12_COMPARISON_FUNC_LESS;
		}
		return D3D12_COMPARISON_FUNC_NEVER;
	}

	uint32_t GfxDeviceDX12::GetSampler(WrapMode wrapMode, FilterMode filterMode)
	{
		size_t key = static_cast<size_t>(wrapMode) << 8 | static_cast<size_t>(filterMode) << 16;
		for (size_t i = 0; i < m_Samplers.size(); ++i)
		{
			if (m_Samplers[i].first == key)
			{
				return static_cast<uint32_t>(i);
			}
		}

		D3D12_TEXTURE_ADDRESS_MODE adress = GetAdressModeD3D12(wrapMode);
		D3D12_FILTER filter = GetFilterD3D12(filterMode);

		D3D12_SAMPLER_DESC samplerDesc = {};
		samplerDesc.Filter = filter;
		samplerDesc.AddressU = adress;
		samplerDesc.AddressV = adress;
		samplerDesc.AddressW = adress;
		samplerDesc.MipLODBias = 0.0f;
		samplerDesc.MaxAnisotropy = filter == D3D12_FILTER_ANISOTROPIC ? 8 : 1;
		samplerDesc.ComparisonFunc = GetComparisonD3D12(filterMode);
		samplerDesc.MinLOD = -FLT_MAX;
		samplerDesc.MaxLOD = FLT_MAX;

		GfxHandleDX12 handle = m_SamplerHeap.AllocatePersistent();
		m_Device->CreateSampler(&samplerDesc, handle.GetCPU());

		GfxHandleDX12 ringHandle = m_SamplerRingHeap.AllocatePersistent();
		m_Device->CopyDescriptorsSimple(1, ringHandle.GetCPU(), handle.GetCPU(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
		
		m_Samplers.push_back(std::make_pair(key, handle));
		return static_cast<uint32_t>(m_Samplers.size() - 1);
	}

	uint32_t GfxDeviceDX12::GetSamplersOffset(const uint8_t* indexes, uint32_t size)
	{
		for (size_t i = 0; i < m_SamplerHeapOffsets.size(); ++i)
		{
			if (memcmp(m_SamplerHeapOffsets[i].data, indexes, sizeof(uint8_t) * size) == 0)
			{
				return m_SamplerHeapOffsets[i].offset;
			}
		}
		auto handle = m_SamplerRingHeap.AllocateTemporary(size);
		uint32_t offset = handle.GetIndex();
		for (size_t i = 0; i < size; ++i)
		{
			m_Device->CopyDescriptorsSimple(1, m_SamplerRingHeap.GetCPU(offset + static_cast<uint32_t>(i)), m_SamplerHeap.GetCPU(indexes[i]), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
		}
		SamplerOffsetData offsetData = {};
		memcpy(offsetData.data, indexes, sizeof(uint8_t) * size);
		offsetData.offset = offset;
		m_SamplerHeapOffsets.push_back(std::move(offsetData));
		return offset;
	}

	void GfxDeviceDX12::ResizeBackbufferIfNeeded()
	{
		if (m_BackbufferResizeRequest.x > 0)
		{
			for (UINT i = 0; i < BUFFER_COUNT; i++)
			{
				BackbufferData& backbuffer = m_Backbuffers[i];
				backbuffer.resource = nullptr;
				backbuffer.renderTargetView.Free();
			}

			HRESULT hr = m_SwapChain->ResizeBuffers(BUFFER_COUNT, m_BackbufferResizeRequest.x, m_BackbufferResizeRequest.y, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "ResizeBuffers failed."));
				return;
			}

			for (UINT i = 0; i < BUFFER_COUNT; i++)
			{
				BackbufferData& backbuffer = m_Backbuffers[i];
				hr = m_SwapChain->GetBuffer(i, IID_PPV_ARGS(&backbuffer.resource));
				if (FAILED(hr))
				{
					BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Error getting swapchain buffer."));
					return;
				}
				GfxHandleDX12 handle = m_RtvHeap.AllocatePersistent();
				m_Device->CreateRenderTargetView(backbuffer.resource.Get(), nullptr, handle.GetCPU());
				backbuffer.renderTargetView = handle;
				backbuffer.state = D3D12_RESOURCE_STATE_PRESENT;
			}

			m_BackbufferIndex = m_SwapChain->GetCurrentBackBufferIndex();
			m_BackbufferResizeRequest = {};
		}
	}
}