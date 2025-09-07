#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class AssetImporterEditor : public ObjectEditor
	{
	public:
		virtual void OnPrepareTargets(const List<Object*>& targets) override;
		virtual void OnEnable() override;
		virtual void OnDisable() override;
		virtual void OnDrawInspector() override;

	private:
		ObjectEditor* m_MainObjectEditor;
	};
}