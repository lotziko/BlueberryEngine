#include "RegisterObjectEditors.h"

#include "ObjectEditorDB.h"

#include "Concrete\EntityEditor.h"
#include "Concrete\TransformEditor.h"
#include "Concrete\CameraEditor.h"
#include "Concrete\SpriteRendererEditor.h"
#include "Concrete\MeshRendererEditor.h"
#include "Concrete\SkyRendererEditor.h"
#include "Concrete\LightEditor.h"
#include "Concrete\SphereColliderEditor.h"
#include "Concrete\BoxColliderEditor.h"
#include "Concrete\CharacterControllerEditor.h"
#include "Concrete\MaterialEditor.h"
#include "Concrete\AssetImporterEditor.h"
#include "Concrete\TextureImporterEditor.h"
#include "Concrete\ModelImporterEditor.h"
#include "Concrete\MeshEditor.h"

namespace Blueberry
{
	void RegisterObjectEditors()
	{
		REGISTER_OBJECT_EDITOR(EntityEditor, Entity);
		REGISTER_OBJECT_EDITOR(TransformEditor, Transform);
		REGISTER_OBJECT_EDITOR(CameraEditor, Camera);
		REGISTER_OBJECT_EDITOR(SpriteRendererEditor, SpriteRenderer);
		REGISTER_OBJECT_EDITOR(MeshRendererEditor, MeshRenderer);
		REGISTER_OBJECT_EDITOR(SkyRendererEditor, SkyRenderer);
		REGISTER_OBJECT_EDITOR(LightEditor, Light);
		REGISTER_OBJECT_EDITOR(SphereColliderEditor, SphereCollider);
		REGISTER_OBJECT_EDITOR(BoxColliderEditor, BoxCollider);
		REGISTER_OBJECT_EDITOR(CharacterControllerEditor, CharacterController);
		REGISTER_OBJECT_EDITOR(MaterialEditor, Material);
		REGISTER_OBJECT_EDITOR(AssetImporterEditor, AssetImporter);
		REGISTER_OBJECT_EDITOR(TextureImporterEditor, TextureImporter);
		REGISTER_OBJECT_EDITOR(ModelImporterEditor, ModelImporter);
		REGISTER_OBJECT_EDITOR(MeshEditor, Mesh);
		REGISTER_OBJECT_EDITOR(ObjectEditor, Object);
	}
}