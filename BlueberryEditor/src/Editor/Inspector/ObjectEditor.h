#pragma once

#include "Blueberry\Core\ClassDB.h"
#include "Editor\Serialization\SerializedProperty.h"
#include "Editor\Serialization\SerializedObject.h"

namespace Blueberry
{
	class Object;
	class Texture;
	class SerializedObject;
	class SerializedProperty;

	class ObjectEditor
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		ObjectEditor() = default;
		virtual ~ObjectEditor() = default;

		virtual Texture* GetIcon(Object* object);
		void DrawScene(Object* object);

		void Enable();
		void DrawInspector();
		void DrawSceneSelected();

		static ObjectEditor* GetEditor(Object* object); // Used for inspector and complex selection gizmos
		static ObjectEditor* GetEditor(const List<Object*>& objects);
		static ObjectEditor* GetDefaultEditor(Object* object); // Used only for simple gizmos
		static void ReleaseEditor(ObjectEditor* editor);

	protected:
		virtual void OnPrepareTargets(const List<Object*>& targets);
		virtual void OnEnable();
		virtual void OnDisable();
		virtual void OnDrawInspector();
		virtual void OnDrawScene();
		virtual void OnDrawSceneSelected();

	private:
		void DrawField(Object* object, FieldInfo& info);

		static Dictionary<ObjectId, ObjectEditor*> s_Editors;
		static Dictionary<size_t, ObjectEditor*> s_DefaultEditors;

	protected:
		std::shared_ptr<SerializedObject> m_SerializedObject;
		Object* m_Object;
		bool m_HasPadding = true;
	};
}