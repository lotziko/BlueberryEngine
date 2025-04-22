#include "bbpch.h"
#include "RegisterEditorTypes.h"

#include "Blueberry\Core\ClassDB.h"

#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\Panels\EditorWindow.h"
#include "Editor\Panels\Project\ProjectBrowser.h"
#include "Editor\Panels\Hierarchy\SceneHierarchy.h"
#include "Editor\Panels\Inspector\SceneInspector.h"
#include "Editor\Panels\Scene\SceneArea.h"
#include "Editor\Panels\Game\GameView.h"
#include "Editor\Panels\Picking\SearchWindow.h"
#include "Editor\Panels\Statistics\StatisticsWindow.h"

namespace Blueberry
{
	void RegisterEditorTypes()
	{
		REGISTER_CLASS(PrefabInstance);
		REGISTER_ABSTRACT_CLASS(EditorWindow);
		REGISTER_CLASS(ProjectBrowser);
		REGISTER_CLASS(SceneHierarchy);
		REGISTER_CLASS(SceneInspector);
		REGISTER_CLASS(SceneArea);
		REGISTER_CLASS(GameView);
		REGISTER_CLASS(SearchWindow);
		REGISTER_CLASS(StatisticsWindow);
	}
}