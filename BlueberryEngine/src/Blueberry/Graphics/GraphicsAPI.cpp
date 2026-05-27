#include "GraphicsAPI.h"

namespace Blueberry
{
	GraphicsAPI::API GraphicsAPI::s_API = GraphicsAPI::API::DX12;

	GraphicsAPI::API GraphicsAPI::GetAPI()
	{
		return s_API;
	}
}