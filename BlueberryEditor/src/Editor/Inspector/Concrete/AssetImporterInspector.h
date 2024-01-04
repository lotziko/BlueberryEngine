#pragma once
#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class AssetImporterInspector : public ObjectInspector
	{
	public:
		virtual void Draw(Object* object) override;
	};
}