#include "bbpch.h"
#include "GfxDevice.h"

#include "GraphicsAPI.h"

#include "Blueberry\Graphics\Mesh.h"

#include "Concrete\DX11\GfxDeviceDX11.h"

namespace Blueberry
{
	GfxDevice* GfxDevice::Create()
	{
		switch (GraphicsAPI::GetAPI())
		{
		case GraphicsAPI::API::None:
			BB_ERROR("API doesn't exist.");
			return nullptr;
		case GraphicsAPI::API::DX11:
			return new GfxDeviceDX11();
		}

		BB_ERROR("API doesn't exist.");
		return nullptr;
	}
}