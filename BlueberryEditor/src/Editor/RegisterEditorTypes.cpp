#include "bbpch.h"
#include "RegisterEditorTypes.h"

#include "Blueberry\Core\ClassDB.h"

#include "Editor\Prefabs\PrefabInstance.h"

namespace Blueberry
{
	void RegisterEditorTypes()
	{
		REGISTER_CLASS(PrefabInstance);
	}
}