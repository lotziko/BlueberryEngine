#include "bbpch.h"
#include "RegisterObjectInspectors.h"

#include "ObjectInspectorDB.h"

#include "Concrete\EntityInspector.h"
#include "Concrete\TransformInspector.h"
#include "Concrete\CameraInspector.h"
#include "Concrete\SpriteRendererInspector.h"

namespace Blueberry
{
	void RegisterObjectInspectors()
	{
		REGISTER_OBJECT_INSPECTOR(EntityInspector, Entity);
		REGISTER_OBJECT_INSPECTOR(TransformInspector, Transform);
		REGISTER_OBJECT_INSPECTOR(CameraInspector, Camera);
		REGISTER_OBJECT_INSPECTOR(SpriteRendererInspector, SpriteRenderer);
	}
}