#pragma once

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	class Texture;

	class ObjectInspector
	{
	public:
		ObjectInspector() = default;
		virtual ~ObjectInspector() = default;

		virtual Texture* GetIcon(Object* object);
		virtual void Draw(Object* object);
		virtual void DrawScene(Object* object);

	private:
		void DrawField(Object* object, FieldInfo& info);
	};
}