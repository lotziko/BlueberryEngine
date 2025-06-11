#include "PostProcessing.h"

#include "AutoExposure.h"
#include "..\Buffers\PostProcessingConstantBuffer.h"

#include "..\..\Assets\AssetLoader.h"
#include "..\GfxDevice.h"
#include "Blueberry\Graphics\Texture2D.h"

namespace Blueberry
{
	static size_t s_BlueNoiseLUTId = TO_HASH("_BlueNoiseLUT");

	void PostProcessing::Initialize()
	{
		if (s_BlueNoiseLUT == nullptr)
		{
			s_BlueNoiseLUT = static_cast<Texture2D*>(AssetLoader::Load("assets/textures/BlueNoiseLUT.png"));
			GfxDevice::SetGlobalTexture(s_BlueNoiseLUTId, s_BlueNoiseLUT->Get());
		}
	}

	void PostProcessing::Draw(GfxTexture* color, const Rectangle& viewport)
	{
		float exposure = 0.2f / AutoExposure::Calculate(color, viewport);
		PostProcessingConstantBuffer::BindData(exposure);
	}
}
