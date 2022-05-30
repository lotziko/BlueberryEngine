#pragma once

#include "Blueberry\Core\Layer.h"
#include "Blueberry\Events\WindowEvent.h"
#include "Panels\SceneHierarchy.h"

class Scene;
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
	void DrawUI();
	void DrawDockSpace();
	void DrawMenuBar();

private:
	Ref<Scene> m_Scene;
	Ref<ImGuiRenderer> m_ImGuiRenderer;
	Ref<Texture> m_BackgroundTexture;

	SceneHierarchy m_SceneHierarchy;
	Vector4 m_Viewport;
};