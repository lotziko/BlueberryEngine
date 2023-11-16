#pragma once

namespace Blueberry
{
	class EditorAssetExporter
	{
	public:
		virtual void Save(const std::string& path, Object* object) = 0;
	};
}