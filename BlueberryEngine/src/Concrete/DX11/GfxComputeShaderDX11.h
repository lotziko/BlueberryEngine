#pragma once

#include "Blueberry\Core\Base.h"
#include "..\..\Blueberry\Graphics\GfxComputeShader.h"
#include "..\Windows\ComPtr.h"
#include "DX11.h"

namespace Blueberry
{
	class GfxComputeShaderDX11 : public GfxComputeShader
	{
	public:
		GfxComputeShaderDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxComputeShaderDX11() final = default;

		bool Initialize(void* computeData);

	private:
		ComPtr<ID3D11ComputeShader> m_ComputeShader = nullptr;
		ComPtr<ID3DBlob> m_ComputeShaderBuffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;

		List<size_t> m_SRVSlots = {};
		List<size_t> m_UAVSlots = {};
		List<size_t> m_ConstantBufferSlots = {};
		List<size_t> m_SamplerSlots = {};
	};
}