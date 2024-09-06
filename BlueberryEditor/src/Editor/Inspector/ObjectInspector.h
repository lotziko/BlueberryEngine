#pragma once

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	class Texture2D;

	class ObjectInspector
	{
	public:
		virtual const char* GetIconPath(Object* object);
		virtual void Draw(Object* object);
		virtual void DrawScene(Object* object);

	private:
		void DrawField(Object* object, FieldInfo& info);
	};
}