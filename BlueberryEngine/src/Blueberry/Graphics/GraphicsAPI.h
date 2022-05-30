#pragma once

class GraphicsAPI
{
public:
	enum class API
	{
		None = 0,
		DX11 = 1,
	};
public:
	inline static API GetAPI()
	{
		return s_API;
	}
private:
	static API s_API;
};