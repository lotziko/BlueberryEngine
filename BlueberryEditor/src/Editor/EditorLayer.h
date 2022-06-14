#pragma once

#include "Blueberry\Core\Layer.h"
#include "Blueberry\Events\WindowEvent.h"

#include "Editor\Panels\Hierarchy\SceneHierarchy.h"
#include "Editor\Panels\Inspector\SceneInspector.h"
#include "Editor\Panels\Scene\SceneArea.h"

class Scene;
class Camera;
class Texture;
class ImGuiRenderer;

class EditorLayer : public Layer
{
public:
	EditorLayer() = default;

	virtual void OnAttach() override;
	virtual void OnDraw() override;
	virtual void OnResizeEvent(const Event& event);

private:
	void DrawDockSpace();
	void DrawMenuBar();

private:
	Ref<Scene> m_Scene;
	Ref<ImGuiRenderer> m_ImGuiRenderer;
	Ref<Texture> m_BackgroundTexture;

	SceneHierarchy m_SceneHierarchy;
	SceneInspector m_SceneInspector;
	SceneArea m_SceneArea;
};