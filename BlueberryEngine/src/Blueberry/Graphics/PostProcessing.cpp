#include "PostProcessing.h"

#include "AutoExposure.h"
#include "PostProcessingConstantBuffer.h"

#include "..\Assets\AssetLoader.h"
#include "GfxDevice.h"
#include "Blueberry\Graphics\Texture2D.h"

namespace Blueberry
{
	static size_t s_BlueNoiseLUTId = TO_HASH("_BlueNoiseLUT");

	void PostProcessing::Draw(GfxTexture* color, const Rectangle& viewport)
	{
		if (m_BlueNoiseLUT == nullptr)
		{
			m_BlueNoiseLUT = static_cast<Texture2D*>(AssetLoader::Load("assets/textures/BlueNoiseLUT.png"));
		}

		float exposure = 0.2f / AutoExposure::Calculate(color, viewport);
		PostProcessingConstantBuffer::BindData(exposure);

		GfxDevice::SetGlobalTexture(s_BlueNoiseLUTId, m_BlueNoiseLUT->Get());
	}
}
