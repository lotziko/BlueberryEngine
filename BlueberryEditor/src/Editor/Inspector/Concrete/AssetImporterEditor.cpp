#include "AssetImporterEditor.h"

#include "Editor\Inspector\ObjectEditorDB.h"
#include "Editor\Assets\AssetImporter.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	void AssetImporterEditor::OnPrepareTargets(const List<Object*>& targets)
	{
		for (Object* target : targets)
		{
			AssetImporter* assetImporter = static_cast<AssetImporter*>(target);
			assetImporter->ImportDataIfNeeded();
		}
	}

	void AssetImporterEditor::OnEnable()
	{
		List<Object*> mainObjects;
		for (Object* target : m_SerializedObject->GetTargets())
		{
			AssetImporter* assetImporter = static_cast<AssetImporter*>(m_SerializedObject->GetTarget());
			assetImporter->ImportDataIfNeeded();
			Object* mainObject = ObjectDB::GetObjectFromGuid(assetImporter->GetGuid(), assetImporter->GetMainObject());
			if (mainObject != nullptr)
			{
				mainObjects.emplace_back(mainObject);
			}
		}
		if (mainObjects.size() > 0)
		{
			m_MainObjectEditor = ObjectEditor::GetEditor(mainObjects);
		}
	}

	void AssetImporterEditor::OnDisable()
	{
		if (m_MainObjectEditor != nullptr)
		{
			ObjectEditor::ReleaseEditor(m_MainObjectEditor);
		}
	}

	void AssetImporterEditor::OnDrawInspector()
	{
		if (m_MainObjectEditor != nullptr)
		{
			m_MainObjectEditor->DrawInspector();
		}
	}
}
