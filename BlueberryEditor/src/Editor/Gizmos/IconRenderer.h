#pragma once

#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Scene;
	class Camera;
	class Texture;
	class Material;
	class Transform;
	class Component;
	class ObjectEditor;

	class IconRenderer
	{
	public:
		struct IconInfo
		{
			ObjectPtr<Transform> transform;
			ObjectPtr<Component> component;
			ObjectEditor* editor;
			float distanceToCamera;
		};

	public:
		static bool Initialize();
		static void Shutdown();
		static void Draw(Scene* scene, Camera* camera);

	private:
		static void ClearCache();
		static void GenerateCache(Scene* scene);

	private:
		static inline Material* s_IconMaterial = nullptr;
		static List<IconInfo> s_IconsCache;
	};
}