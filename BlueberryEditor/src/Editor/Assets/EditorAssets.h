#pragma once

namespace Blueberry
{
	class Object;

	class EditorAssets
	{
	public:
		static Object* Load(const std::string& path);
	};
}