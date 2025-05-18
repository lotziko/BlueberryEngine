#include "RegisterObjectInspectors.h"

#include "ObjectInspectorDB.h"

#include "Concrete\EntityInspector.h"
#include "Concrete\TransformInspector.h"
#include "Concrete\CameraInspector.h"
#include "Concrete\SpriteRendererInspector.h"
#include "Concrete\MeshRendererInspector.h"
#include "Concrete\LightInspector.h"
#include "Concrete\SphereColliderInspector.h"
#include "Concrete\BoxColliderInspector.h"
#include "Concrete\CharacterControllerInspector.h"
#include "Concrete\MaterialInspector.h"
#include "Concrete\AssetImporterInspector.h"
#include "Concrete\TextureImporterInspector.h"
#include "Concrete\ModelImporterInspector.h"
#include "Concrete\MeshInspector.h"

namespace Blueberry
{
	void RegisterObjectInspectors()
	{
		REGISTER_OBJECT_INSPECTOR(EntityInspector, Entity);
		REGISTER_OBJECT_INSPECTOR(TransformInspector, Transform);
		REGISTER_OBJECT_INSPECTOR(CameraInspector, Camera);
		REGISTER_OBJECT_INSPECTOR(SpriteRendererInspector, SpriteRenderer);
		REGISTER_OBJECT_INSPECTOR(MeshRendererInspector, MeshRenderer);
		REGISTER_OBJECT_INSPECTOR(LightInspector, Light);
		REGISTER_OBJECT_INSPECTOR(SphereColliderInspector, SphereCollider);
		REGISTER_OBJECT_INSPECTOR(BoxColliderInspector, BoxCollider);
		REGISTER_OBJECT_INSPECTOR(CharacterControllerInspector, CharacterController);
		REGISTER_OBJECT_INSPECTOR(MaterialInspector, Material);
		REGISTER_OBJECT_INSPECTOR(AssetImporterInspector, AssetImporter);
		REGISTER_OBJECT_INSPECTOR(TextureImporterInspector, TextureImporter);
		REGISTER_OBJECT_INSPECTOR(ModelImporterInspector, ModelImporter);
		REGISTER_OBJECT_INSPECTOR(MeshInspector, Mesh);
		REGISTER_OBJECT_INSPECTOR(ObjectInspector, Object);
	}
}