#include "bbpch.h"
#include "GraphicsDevice.h"

#include "GraphicsAPI.h"

#include "Concrete\DX11\DX11GraphicsDevice.h"

namespace Blueberry
{
	GraphicsDevice* GraphicsDevice::Create()
	{
		switch (GraphicsAPI::GetAPI())
		{
		case GraphicsAPI::API::None:
			BB_ERROR("API doesn't exist.");
			return nullptr;
		case GraphicsAPI::API::DX11:
			return new DX11GraphicsDevice();
		}

		BB_ERROR("API doesn't exist.");
		return nullptr;
	}
}