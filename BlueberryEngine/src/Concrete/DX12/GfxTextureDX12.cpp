#include "GfxTextureDX12.h"

#include "..\Windows\WindowsHelper.h"
#include "..\Windows\DxgiHelper.h"
#include "Blueberry\Tools\StringHelper.h"
#include "GfxDeviceDX12.h"

namespace Blueberry
{
	GfxPointerCache<GfxTextureDX12> GfxTextureDX12::s_PointerCache = {};

	GfxTextureDX12::GfxTextureDX12(GfxDeviceDX12* device) : m_Device(device->GetDevice()), m_GfxDevice(device)
	{
		m_Index = s_PointerCache.Allocate(this);
	}

	GfxTextureDX12::~GfxTextureDX12()
	{
		s_PointerCache.Deallocate(m_Index);
		m_ShaderResourceView.Free();
		m_RenderTargetView.Free();
		m_DepthStencilView.Free();
		m_UnorderedAccessView.Free();
		for (auto& sliceRenderTargetView : m_SlicesRenderTargetViews)
		{
			sliceRenderTargetView.Free();
		}
		m_GfxDevice->Release(m_Resource);
	}

	uint32_t GetQualityLevel(ID3D12Device* device, DXGI_FORMAT format, uint32_t antiAliasing)
	{
		if (antiAliasing > 1)
		{
			D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS multisampleLevels = {};
			multisampleLevels.Format = format;
			multisampleLevels.SampleCount = antiAliasing;
			multisampleLevels.Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE;

			uint32_t qualityLevels = 1;
			HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &qualityLevels, sizeof(qualityLevels));
			return qualityLevels - 1;
		}
		return 0;
	}

	bool GfxTextureDX12::Initialize(const TextureProperties& properties)
	{
		m_Format = static_cast<DXGI_FORMAT>(properties.format);
		m_Width = std::max(properties.width, 1u);
		m_Height = std::max(properties.height, 1u);
		m_Depth = properties.depth;
		m_Dimension = properties.dimension;
		m_FilterMode = properties.filterMode;
		m_WrapMode = properties.wrapMode;
		m_AntiAliasing = std::max(1u, properties.antiAliasing);
		m_Quality = GetQualityLevel(m_Device, m_Format, m_AntiAliasing);
		m_ArraySize = DxgiHelper::GetArraySize(properties.dimension, properties.depth);
		m_MipLevels = std::max(1u, properties.mipCount);

		if (properties.data != nullptr)
		{
			List<D3D12_SUBRESOURCE_DATA> subresourceDatas;
			GatherSubresources(properties.data, subresourceDatas);
			bool result = Initialize(subresourceDatas.data(), static_cast<uint32_t>(subresourceDatas.size()), properties);
			return result;
		}
		else
		{
			return Initialize(nullptr, 0, properties);
		}
	}

	ID3D12Resource* GfxTextureDX12::GetResource() const
	{
		return m_Resource.Get();
	}

	const GfxHandleDX12& GfxTextureDX12::GetSRV() const
	{
		return m_ShaderResourceView;
	}

	const GfxHandleDX12& GfxTextureDX12::GetRTV() const
	{
		return m_RenderTargetView;
	}

	const GfxHandleDX12& GfxTextureDX12::GetRTV(uint32_t arraySlice, uint32_t mipSlice)
	{
		uint32_t index = arraySlice * m_MipLevels + mipSlice;
		const GfxHandleDX12& handle = m_SlicesRenderTargetViews[index];
		if (handle.IsInvalid())
		{
			D3D12_RENDER_TARGET_VIEW_DESC renderTargetViewDesc = {};
			renderTargetViewDesc.Format = m_Format;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				if (m_AntiAliasing <= 1)
				{
					renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
					renderTargetViewDesc.Texture2D.MipSlice = mipSlice;
					renderTargetViewDesc.Texture2D.PlaneSlice = 0;
				}
				break;
			case TextureDimension::Texture2DArray:
				if (m_AntiAliasing > 1)
				{
					renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
					renderTargetViewDesc.Texture2DMSArray.FirstArraySlice = arraySlice;
					renderTargetViewDesc.Texture2DMSArray.ArraySize = m_ArraySize;
				}
				else
				{
					renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
					renderTargetViewDesc.Texture2DArray.MipSlice = mipSlice;
					renderTargetViewDesc.Texture2DArray.FirstArraySlice = arraySlice;
					renderTargetViewDesc.Texture2DArray.ArraySize = m_ArraySize;
					renderTargetViewDesc.Texture2DArray.PlaneSlice = 0;
				}
				break;
			case TextureDimension::TextureCube:
				renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
				renderTargetViewDesc.Texture2DArray.MipSlice = mipSlice;
				renderTargetViewDesc.Texture2DArray.FirstArraySlice = arraySlice;
				renderTargetViewDesc.Texture2DArray.ArraySize = 6;
				renderTargetViewDesc.Texture2DArray.PlaneSlice = 0;
				break;
			case TextureDimension::Texture3D:
				renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
				renderTargetViewDesc.Texture3D.MipSlice = mipSlice;
				renderTargetViewDesc.Texture3D.FirstWSlice = arraySlice;
				renderTargetViewDesc.Texture3D.WSize = 1;
				break;
			}

			m_GfxDevice->CreateRTV(m_Resource.Get(), &renderTargetViewDesc, &m_SlicesRenderTargetViews[index]);
			return m_SlicesRenderTargetViews[index];
		}
		return handle;
	}

	D3D12_RESOURCE_STATES GfxTextureDX12::GetState() const
	{
		return m_State;
	}

	uint32_t GfxTextureDX12::GetWidth() const
	{
		return m_Width;
	}

	uint32_t GfxTextureDX12::GetHeight() const
	{
		return m_Height;
	}

	TextureFormat GfxTextureDX12::GetFormat() const
	{
		return static_cast<TextureFormat>(m_Format);
	}

	void* GfxTextureDX12::GetHandle()
	{
		uint64_t generation = m_GfxDevice->GetGeneration();
		if (generation > m_HandleGeneration)
		{
			uint32_t descriptorCount = 1;
			m_HandleShaderResourceView = m_GfxDevice->GetCbvSrvDsvRingHeap().AllocateTemporary(descriptorCount);
			m_Device->CopyDescriptors(1, &m_HandleShaderResourceView.GetCPU(), &descriptorCount, descriptorCount, &m_ShaderResourceView.GetCPU(), nullptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			m_HandleGeneration = generation;
		}
		return reinterpret_cast<void*>(m_HandleShaderResourceView.GetGPU().ptr);
	}

	void GfxTextureDX12::GetData(void* data, const Rectangle& area)
	{
		if (data != nullptr)
		{
			ID3D12GraphicsCommandList* commandList = m_GfxDevice->GetCommandList();
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), m_State, D3D12_RESOURCE_STATE_COPY_SOURCE));
			m_GfxDevice->GetReadbackBuffer().ReadTexture(m_Resource.Get(), data, area);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, m_State));
		}
	}

	void GfxTextureDX12::GetData(void* data)
	{
		if (data != nullptr)
		{
			ID3D12GraphicsCommandList* commandList = m_GfxDevice->GetCommandList();
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), m_State, D3D12_RESOURCE_STATE_COPY_SOURCE));
			m_GfxDevice->GetReadbackBuffer().ReadTexture(m_Resource.Get(), data, m_ArraySize * m_MipLevels);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, m_State));
		}
	}

	void GfxTextureDX12::SetData(void* data, size_t size)
	{
		List<D3D12_SUBRESOURCE_DATA> subresourceDatas;
		GatherSubresources(data, subresourceDatas);
		ID3D12GraphicsCommandList* commandList = m_GfxDevice->GetCommandList();
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), m_State, D3D12_RESOURCE_STATE_COPY_DEST));
		m_GfxDevice->GetUploadBuffer().UploadTexture(m_Resource.Get(), subresourceDatas.data(), static_cast<UINT>(subresourceDatas.size()), 512ull);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, m_State));
	}

	void GfxTextureDX12::SetWrapMode(WrapMode wrapMode)
	{
		if (m_WrapMode != wrapMode)
		{
			m_WrapMode = wrapMode;
			m_Sampler = UINT8_MAX;
		}
	}

	void GfxTextureDX12::SetFilterMode(FilterMode filterMode)
	{
		if (m_FilterMode != filterMode)
		{
			m_FilterMode = filterMode;
			m_Sampler = UINT8_MAX;
		}
	}

	void GfxTextureDX12::SetName(const String& name)
	{
		m_Resource->SetName(StringHelper::StringToWide(name).c_str());
	}

	D3D12_RESOURCE_DIMENSION GetResourceDimension(TextureDimension dimension)
	{
		switch (dimension)
		{
		case TextureDimension::Texture2D:
		case TextureDimension::Texture2DArray:
		case TextureDimension::TextureCube:
		case TextureDimension::TextureCubeArray:
			return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		case TextureDimension::Texture3D:
			return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
		default:
			return D3D12_RESOURCE_DIMENSION_UNKNOWN;
		}
	}
	
	void GfxTextureDX12::GatherSubresources(const void* data, List<D3D12_SUBRESOURCE_DATA>& subresourceDatas)
	{
		uint32_t bitsPerPixel = DxgiHelper::GetBitsPerPixel(m_Format);
		uint32_t size = m_ArraySize * m_MipLevels;
		subresourceDatas.resize(size);

		const uint8_t* ptr = static_cast<const uint8_t*>(data);
		if (DxgiHelper::IsCompressed(m_Format))
		{
			uint32_t blockSize = bitsPerPixel * 16 / 8;
			for (uint32_t i = 0; i < m_ArraySize; ++i)
			{
				uint32_t width = m_Width;
				uint32_t height = m_Height;
				for (uint32_t j = 0; j < m_MipLevels; ++j)
				{
					UINT subresource = D3D12CalcSubresource(j, i, 0, m_MipLevels, m_ArraySize);
					D3D12_SUBRESOURCE_DATA subresourceData = {};
					subresourceData.pData = ptr;
					subresourceData.RowPitch = static_cast<size_t>(std::max(1u, (width + 3) / 4) * blockSize);
					subresourceData.SlicePitch = 0;
					subresourceDatas[subresource] = subresourceData;
					ptr += subresourceData.RowPitch * std::max(1u, (height + 3) / 4);
					width /= 2;
					height /= 2;
				}
			}
		}
		else
		{
			for (uint32_t i = 0; i < m_ArraySize; ++i)
			{
				uint32_t width = m_Width;
				uint32_t height = m_Height;
				for (uint32_t j = 0; j < m_MipLevels; ++j)
				{
					UINT subresource = D3D12CalcSubresource(j, i, 0, m_MipLevels, m_ArraySize);
					D3D12_SUBRESOURCE_DATA subresourceData = {};
					subresourceData.pData = ptr;
					subresourceData.RowPitch = static_cast<size_t>(width * bitsPerPixel / 8);
					subresourceData.SlicePitch = m_Depth > 0 ? (subresourceData.RowPitch * height) : 0;
					subresourceDatas[subresource] = subresourceData;
					ptr += width * height * bitsPerPixel / 8;
					width /= 2;
					height /= 2;
				}
			}
		}
	}

	bool GfxTextureDX12::Initialize(D3D12_SUBRESOURCE_DATA* subresourceData, uint32_t subresourceCount, const TextureProperties& properties)
	{
		bool useDSV = DxgiHelper::IsDepth(m_Format);
		bool useRTV = !useDSV && HasFlag(properties.usageFlags, TextureUsageFlags::RenderTarget);
		bool useUAV = HasFlag(properties.usageFlags, TextureUsageFlags::UnorderedAccess);
		bool isReadable = HasFlag(properties.usageFlags, TextureUsageFlags::CPUReadable);
		bool isWritable = HasFlag(properties.usageFlags, TextureUsageFlags::CPUWritable);
		bool useStaging = isReadable || isWritable;

		D3D12_RESOURCE_DESC resourceDesc = {};
		resourceDesc.MipLevels = m_MipLevels;
		resourceDesc.Format = m_Format;
		resourceDesc.Width = m_Width;
		resourceDesc.Height = m_Height;
		resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
		resourceDesc.DepthOrArraySize = m_Dimension == TextureDimension::Texture3D ? m_Depth : m_ArraySize;
		resourceDesc.SampleDesc.Count = m_AntiAliasing;
		resourceDesc.SampleDesc.Quality = m_Quality;
		resourceDesc.Dimension = GetResourceDimension(m_Dimension);

		if (useRTV)
		{
			resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
		}

		if (useDSV)
		{
			resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
		}

		if (useUAV)
		{
			resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		}

		D3D12_CLEAR_VALUE clearValue = {};
		clearValue.Format = m_Format;
		if (useDSV)
		{
			clearValue.DepthStencil.Depth = 1.0f;
		}

		D3D12_RESOURCE_STATES targetState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
		D3D12_RESOURCE_STATES initialState = subresourceData == nullptr ? targetState : D3D12_RESOURCE_STATE_COPY_DEST;
		HRESULT hr = m_Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc, initialState, (useRTV || useDSV) ? &clearValue : nullptr, IID_PPV_ARGS(&m_Resource));
		
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create texture."));
			return false;
		}
		m_State = targetState;

		if (subresourceData != nullptr)
		{
			ID3D12GraphicsCommandList* commandList = m_GfxDevice->GetCommandList();
			m_GfxDevice->GetUploadBuffer().UploadTexture(m_Resource.Get(), subresourceData, subresourceCount, 512ull);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), initialState, targetState));
		}

		// SRV
		D3D12_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc = {};
		shaderResourceViewDesc.Format = DxgiHelper::GetSRVFormat(resourceDesc.Format);
		shaderResourceViewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

		switch (m_Dimension)
		{
		case TextureDimension::Texture2D:
			shaderResourceViewDesc.ViewDimension = m_AntiAliasing > 1 ? D3D12_SRV_DIMENSION_TEXTURE2DMS : D3D12_SRV_DIMENSION_TEXTURE2D;
			shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
			shaderResourceViewDesc.Texture2D.PlaneSlice = 0;
			shaderResourceViewDesc.Texture2D.MipLevels = m_MipLevels;
			shaderResourceViewDesc.Texture2D.ResourceMinLODClamp = 0.0f;
			break;
		case TextureDimension::Texture2DArray:
			if (m_AntiAliasing > 1)
			{
				shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
				shaderResourceViewDesc.Texture2DMSArray.FirstArraySlice = 0;
				shaderResourceViewDesc.Texture2DMSArray.ArraySize = m_ArraySize;
			}
			else
			{
				shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
				shaderResourceViewDesc.Texture2DArray.MipLevels = m_MipLevels;
				shaderResourceViewDesc.Texture2DArray.FirstArraySlice = 0;
				shaderResourceViewDesc.Texture2DArray.ArraySize = m_ArraySize;
				shaderResourceViewDesc.Texture2DArray.PlaneSlice = 0;
				shaderResourceViewDesc.Texture2DArray.ResourceMinLODClamp = 0.0f;
			}
			break;
		case TextureDimension::TextureCube:
			shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
			shaderResourceViewDesc.TextureCube.MipLevels = m_MipLevels;
			shaderResourceViewDesc.TextureCube.MostDetailedMip = 0;
			shaderResourceViewDesc.TextureCube.ResourceMinLODClamp = 0.0f;
			break;
		case TextureDimension::TextureCubeArray:
			shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
			shaderResourceViewDesc.TextureCubeArray.MipLevels = m_MipLevels;
			shaderResourceViewDesc.TextureCubeArray.MostDetailedMip = 0;
			shaderResourceViewDesc.TextureCubeArray.First2DArrayFace = 0;
			shaderResourceViewDesc.TextureCubeArray.NumCubes = m_ArraySize / 6;
			shaderResourceViewDesc.TextureCubeArray.ResourceMinLODClamp = 0.0f;
			break;
		case TextureDimension::Texture3D:
			shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
			shaderResourceViewDesc.Texture3D.MipLevels = m_MipLevels;
			shaderResourceViewDesc.Texture3D.MostDetailedMip = 0;
			shaderResourceViewDesc.Texture3D.ResourceMinLODClamp = 0.0f;
			break;
		}
		
		m_GfxDevice->CreateSRV(m_Resource.Get(), &shaderResourceViewDesc, &m_ShaderResourceView);

		// RTV
		if (useRTV)
		{
			D3D12_RENDER_TARGET_VIEW_DESC renderTargetViewDesc = {};
			renderTargetViewDesc.Format = m_Format;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				renderTargetViewDesc.ViewDimension = m_AntiAliasing > 1 ? D3D12_RTV_DIMENSION_TEXTURE2DMS : D3D12_RTV_DIMENSION_TEXTURE2D;
				renderTargetViewDesc.Texture2D.MipSlice = 0;
				renderTargetViewDesc.Texture2D.PlaneSlice = 0;
				m_SlicesRenderTargetViews.resize(m_MipLevels);
				break;
			case TextureDimension::Texture2DArray:
				if (m_AntiAliasing > 1)
				{
					renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
					renderTargetViewDesc.Texture2DMSArray.FirstArraySlice = 0;
					renderTargetViewDesc.Texture2DMSArray.ArraySize = m_ArraySize;
				}
				else
				{
					renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
					renderTargetViewDesc.Texture2DArray.MipSlice = 0;
					renderTargetViewDesc.Texture2DArray.FirstArraySlice = 0;
					renderTargetViewDesc.Texture2DArray.ArraySize = m_ArraySize;
					renderTargetViewDesc.Texture2DArray.PlaneSlice = 0;
				}
				m_SlicesRenderTargetViews.resize(static_cast<size_t>(m_ArraySize* m_MipLevels));
				break;
			case TextureDimension::TextureCube:
				renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
				renderTargetViewDesc.Texture2DArray.MipSlice = 0;
				renderTargetViewDesc.Texture2DArray.FirstArraySlice = 0;
				renderTargetViewDesc.Texture2DArray.ArraySize = 6;
				renderTargetViewDesc.Texture2DArray.PlaneSlice = 0;
				m_SlicesRenderTargetViews.resize(static_cast<size_t>(6 * m_MipLevels));
				break;
			case TextureDimension::Texture3D:
				renderTargetViewDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
				renderTargetViewDesc.Texture3D.MipSlice = 0;
				renderTargetViewDesc.Texture3D.FirstWSlice = 0;
				renderTargetViewDesc.Texture3D.WSize = -1;
				m_SlicesRenderTargetViews.resize(static_cast<size_t>(properties.depth* m_MipLevels));
				break;
			}
			
			m_GfxDevice->CreateRTV(m_Resource.Get(), &renderTargetViewDesc, &m_RenderTargetView);
		}

		// DSV
		if (useDSV)
		{
			D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc = {};
			depthStencilViewDesc.Format = m_Format;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				depthStencilViewDesc.ViewDimension = m_AntiAliasing > 1 ? D3D12_DSV_DIMENSION_TEXTURE2DMS : D3D12_DSV_DIMENSION_TEXTURE2D;
				depthStencilViewDesc.Texture2D.MipSlice = 0;
				break;
			case TextureDimension::Texture2DArray:
				if (m_AntiAliasing > 1)
				{
					depthStencilViewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
					depthStencilViewDesc.Texture2DMSArray.FirstArraySlice = 0;
					depthStencilViewDesc.Texture2DMSArray.ArraySize = m_ArraySize;
				}
				else
				{
					depthStencilViewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
					depthStencilViewDesc.Texture2DArray.MipSlice = 0;
					depthStencilViewDesc.Texture2DArray.FirstArraySlice = 0;
					depthStencilViewDesc.Texture2DArray.ArraySize = m_ArraySize;
				}
				break;
			}

			m_GfxDevice->CreateDSV(m_Resource.Get(), &depthStencilViewDesc, &m_DepthStencilView);
		}

		// UAV
		if (useUAV)
		{
			D3D12_UNORDERED_ACCESS_VIEW_DESC unorderedAccessViewDesc = {};
			unorderedAccessViewDesc.Format = m_Format;

			switch (m_Dimension)
			{
			case TextureDimension::Texture2D:
				unorderedAccessViewDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
				unorderedAccessViewDesc.Texture2D.MipSlice = 0;
				unorderedAccessViewDesc.Texture2D.PlaneSlice = 0;
				break;
			case TextureDimension::Texture2DArray:
				unorderedAccessViewDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
				unorderedAccessViewDesc.Texture2DArray.MipSlice = 0;
				unorderedAccessViewDesc.Texture2DArray.FirstArraySlice = 0;
				unorderedAccessViewDesc.Texture2DArray.ArraySize = m_ArraySize;
				unorderedAccessViewDesc.Texture2DArray.PlaneSlice = 0;
				break;
			case TextureDimension::Texture3D:
				unorderedAccessViewDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
				unorderedAccessViewDesc.Texture3D.MipSlice = 0;
				unorderedAccessViewDesc.Texture3D.FirstWSlice = 0;
				unorderedAccessViewDesc.Texture3D.WSize = -1;
				break;
			}
			
			m_GfxDevice->CreateUAV(m_Resource.Get(), &unorderedAccessViewDesc, &m_UnorderedAccessView);
		}

		return true;
	}
}