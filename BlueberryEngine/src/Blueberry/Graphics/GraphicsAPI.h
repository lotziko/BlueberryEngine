#pragma once

namespace Blueberry
{
	class GraphicsAPI
	{
	public:
		enum class API
		{
			None = 0,
			DX11 = 1,
		};

	public:
		static API GetAPI();

	private:
		static API s_API;
	};
}