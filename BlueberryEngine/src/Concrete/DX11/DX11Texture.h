#pragma once
#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class DX11Texture : public Texture
	{
	public:
		DX11Texture(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~DX11Texture() final = default;

		bool Initialize(const std::string& path);

		virtual void* GetHandle() override;

		virtual void Bind(const UINT& slot) override;

	private:
		ComRef<ID3D11Texture2D> m_Texture;
		ComRef<ID3D11ShaderResourceView> m_ResourceView;
		ComRef<ID3D11SamplerState> m_SamplerState;

		int m_Width;
		int m_Height;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;
	};
}