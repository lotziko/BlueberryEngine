#include "SceneObjectPicker.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\MeshRenderer.h"
#include "Blueberry\Scene\Components\SkinnedMeshRenderer.h"
#include "Blueberry\Scene\Components\SpriteRenderer.h"
#include "Blueberry\Graphics\Renderer2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Graphics\Buffers\PerDrawDataConstantBuffer.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\Skinning.h"

#include "Editor\Selection.h"
#include "Editor\Inspector\ObjectEditor.h"
#include "Editor\Inspector\ObjectEditorDB.h"
#include "Editor\Misc\SceneHelper.h"

#include "PerObjectDataConstantBuffer.h"

namespace Blueberry
{
	GfxTexture* SceneObjectPicker::s_SceneRenderTarget = nullptr;
	GfxTexture* SceneObjectPicker::s_SceneDepthStencil = nullptr;
	Material* SceneObjectPicker::s_SpriteObjectPickerMaterial = nullptr;
	Material* SceneObjectPicker::s_MeshObjectPickerMaterial = nullptr;
	Material* SceneObjectPicker::s_IconObjectPickerMaterial = nullptr;
	Material* SceneObjectPicker::s_ObjectPickerOutlineMaterial = nullptr;

	static size_t s_PickingTextureId = TO_HASH("_PickingTexture");

	Color ConvertIndexToColor(uint16_t index)
	{
		Color result;
		float packedIndex[2] = { (float)((index >> 8) & 255) / 255.0f, (float)(index & 255) / 255.0f };
		memcpy(&result, packedIndex, sizeof(float) * 2);
		return result;
	}

	void SceneObjectPicker::Initialize()
	{
		TextureProperties textureProperties = {};
		textureProperties.width = Screen::GetWidth();
		textureProperties.height = Screen::GetHeight();
		textureProperties.format = TextureFormat::R8G8B8A8_UNorm;
		textureProperties.usageFlags = TextureUsageFlags::RenderTarget | TextureUsageFlags::CPUReadable;
		GfxDevice::CreateTexture(textureProperties, s_SceneRenderTarget);

		textureProperties.usageFlags = TextureUsageFlags::RenderTarget;
		textureProperties.format = TextureFormat::D24_UNorm;
		GfxDevice::CreateTexture(textureProperties, s_SceneDepthStencil);

		s_SpriteObjectPickerMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/SpriteObjectPicker.shader")));
		s_MeshObjectPickerMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/MeshObjectPicker.shader")));
		s_IconObjectPickerMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/IconObjectPicker.shader")));
		s_ObjectPickerOutlineMaterial = Material::Create(static_cast<Shader*>(AssetLoader::Load("assets/shaders/ObjectPickerOutline.shader")));
	}

	void SceneObjectPicker::Shutdown()
	{
		delete s_SceneRenderTarget;
		delete s_SceneDepthStencil;
		Object::Destroy(s_SpriteObjectPickerMaterial);
		Object::Destroy(s_MeshObjectPickerMaterial);
		Object::Destroy(s_IconObjectPickerMaterial);
		Object::Destroy(s_ObjectPickerOutlineMaterial);
	}

	Object* SceneObjectPicker::Pick(Scene* scene, Camera* camera, int positionX, int positionY)
	{
		if (scene == nullptr)
		{
			return nullptr;
		}

		Ray ray = SceneHelper::ScreenPointToWorldRay(Vector2(static_cast<float>(positionX), static_cast<float>(positionY)), camera);
		float distance = 0;
		Rectangle area = Rectangle(std::min(std::max(positionX, 0), static_cast<int>(camera->GetPixelSize().x)), std::min(std::max(positionY, 0), static_cast<int>(camera->GetPixelSize().y)), 1, 1);
		unsigned char pixel[4];
		Dictionary<int, ObjectId> validObjects;
		uint32_t index = 1;

		GfxDevice::SetRenderTarget(s_SceneRenderTarget, s_SceneDepthStencil);
		GfxDevice::SetViewport(0, 0, static_cast<int>(camera->GetPixelSize().x), static_cast<int>(camera->GetPixelSize().y));
		GfxDevice::ClearColor(Color(0.0f, 0.0f, 0.0f, 0.0f));
		GfxDevice::ClearDepth(1.0f);
		Renderer2D::Begin();
		for (SpriteRenderer* spriteRenderer : scene->GetComponents<SpriteRenderer>())
		{
			Entity* entity = spriteRenderer->GetEntity();
			if (entity->IsActiveInHierarchy())
			{
				if (spriteRenderer->GetTexture() != nullptr && ray.Intersects(spriteRenderer->GetBounds(), distance))
				{
					Renderer2D::Draw(spriteRenderer->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), s_SpriteObjectPickerMaterial, ConvertIndexToColor(index), spriteRenderer->GetSortingOrder());
					validObjects[index] = entity->GetObjectId();
					++index;
				}
			}
		}
		Renderer2D::End();

		for (MeshRenderer* meshRenderer : scene->GetComponents<MeshRenderer>())
		{
			Entity* entity = meshRenderer->GetEntity();
			Mesh* mesh = meshRenderer->GetMesh();
			if (mesh != nullptr && ray.Intersects(meshRenderer->GetBounds(), distance))
			{
				PerDrawDataConstantBuffer::BindData(meshRenderer->GetLocalToWorldMatrix());
				PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
				GfxDevice::Draw(GfxDrawingOperation(mesh, s_MeshObjectPickerMaterial));
				validObjects[index] = entity->GetObjectId();
				++index;
			}
		}

		for (SkinnedMeshRenderer* skinnedMeshRenderer : scene->GetComponents<SkinnedMeshRenderer>())
		{
			Entity* entity = skinnedMeshRenderer->GetEntity();
			Mesh* mesh = skinnedMeshRenderer->GetMesh();
			if (mesh != nullptr && ray.Intersects(skinnedMeshRenderer->GetBounds(), distance))
			{
				PerDrawDataConstantBuffer::BindData(skinnedMeshRenderer->GetLocalToWorldMatrix());
				PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
				GfxDevice::Draw(GfxDrawingOperation(mesh, Skinning::GetVertexBuffer(skinnedMeshRenderer), s_MeshObjectPickerMaterial));
				validObjects[index] = entity->GetObjectId();
				++index;
			}
		}

		Vector3 cameraDirection = Vector3::Transform(Vector3::Forward, camera->GetTransform()->GetRotation());
		for (auto& pair : scene->GetEntities())
		{
			Entity* entity = pair.second.Get();
			if (entity->IsActiveInHierarchy())
			{
				for (uint32_t i = 0; i < entity->GetComponentCount(); ++i)
				{
					Component* component = entity->GetComponentAt(i);
					ObjectEditor* editor = ObjectEditor::GetDefaultEditor(component);
					Texture* icon;
					if ((icon = editor->GetIcon(component)) != nullptr)
					{
						s_IconObjectPickerMaterial->SetTexture("_BaseMap", icon);
						Vector3 position = entity->GetTransform()->GetPosition();
						Matrix modelMatrix = Matrix::CreateScale(-0.75f, 0.75f, 0.75f) * Matrix::CreateBillboard(position, position + cameraDirection, Vector3(0, 1, 0));
						PerDrawDataConstantBuffer::BindData(modelMatrix);
						PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(index));
						GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_IconObjectPickerMaterial));
						validObjects[index] = entity->GetObjectId();
						++index;
						break;
					}
				}
			}
		}

		GfxDevice::SetRenderTarget(nullptr);
		s_SceneRenderTarget->GetData(pixel, area);

		if (pixel[0] > 0 || pixel[1] > 0)
		{
			uint16_t index = (pixel[0] << 8) + (pixel[1]);
			return ObjectDB::GetObject(validObjects[index]);
		}
		else
		{
			return nullptr;
		}
	}

	void SceneObjectPicker::DrawOutline(Scene* scene, Camera* camera, GfxTexture* colorRenderTarget, GfxTexture* depthStencilRenderTarget, const Vector2Int& viewport)	// TODO handle OnSelectionChanged
	{
		if (scene == nullptr)
		{
			return;
		}

		HashSet<ObjectId> outlineEntities;
		List<Transform*> pendingTransforms;
		for (Object* object : Selection::GetActiveObjects())
		{
			if (object != nullptr && object->IsClassType(Entity::Type))
			{
				Entity* entity = static_cast<Entity*>(object);
				if (entity->GetScene() == scene)
				{
					pendingTransforms.push_back(entity->GetTransform());
				}
			}
		}

		while (!pendingTransforms.empty())
		{
			Transform* transform = pendingTransforms.back();
			pendingTransforms.pop_back();

			if (transform == nullptr)
			{
				continue;
			}

			Entity* entity = transform->GetEntity();
			if (!outlineEntities.insert(entity->GetObjectId()).second)
			{
				continue;
			}

			for (const auto& child : transform->GetChildren())
			{
				pendingTransforms.push_back(child.Get());
			}
		}

		GfxDevice::SetRenderTarget(s_SceneRenderTarget);
		GfxDevice::SetViewport(0, 0, viewport.x, viewport.y);
		GfxDevice::ClearColor(Color(0.0f, 0.0f, 0.0f, 0.0f));
		GfxDevice::ClearDepth(1.0f);

		Renderer2D::Begin();
		for (SpriteRenderer* spriteRenderer : scene->GetComponents<SpriteRenderer>())
		{
			Entity* entity = spriteRenderer->GetEntity();
			if (outlineEntities.count(entity->GetObjectId()) > 0 &&	entity->IsActiveInHierarchy())
			{
				if (spriteRenderer->GetTexture() != nullptr)
				{
					Renderer2D::Draw(spriteRenderer->GetLocalToWorldMatrix(), spriteRenderer->GetTexture(), s_SpriteObjectPickerMaterial, ConvertIndexToColor(65535), spriteRenderer->GetSortingOrder());
				}
			}
		}
		Renderer2D::End();

		for (MeshRenderer* meshRenderer : scene->GetComponents<MeshRenderer>())
		{
			Entity* entity = meshRenderer->GetEntity();
			if (outlineEntities.count(entity->GetObjectId()) > 0 && entity->IsActiveInHierarchy())
			{
				Mesh* mesh = meshRenderer->GetMesh();
				if (mesh != nullptr)
				{
					PerDrawDataConstantBuffer::BindData(meshRenderer->GetLocalToWorldMatrix());
					PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(10000));
					GfxDevice::Draw(GfxDrawingOperation(mesh, s_MeshObjectPickerMaterial));
				}
			}
		}

		for (SkinnedMeshRenderer* skinnedMeshRenderer : scene->GetComponents<SkinnedMeshRenderer>())
		{
			Entity* entity = skinnedMeshRenderer->GetEntity();
			if (outlineEntities.count(entity->GetObjectId()) > 0 && entity->IsActiveInHierarchy())
			{
				if (skinnedMeshRenderer->HasRoot())
				{
					Mesh* mesh = skinnedMeshRenderer->GetMesh();
					if (mesh != nullptr)
					{
						PerDrawDataConstantBuffer::BindData(skinnedMeshRenderer->GetLocalToWorldMatrix());
						PerObjectDataConstantBuffer::BindData(ConvertIndexToColor(10000));
						GfxDevice::Draw(GfxDrawingOperation(mesh, Skinning::GetVertexBuffer(skinnedMeshRenderer), s_MeshObjectPickerMaterial));
					}
				}
			}
		}
		
		GfxDevice::SetRenderTarget(colorRenderTarget, depthStencilRenderTarget);
		GfxDevice::SetViewport(0, 0, s_SceneRenderTarget->GetWidth(), s_SceneRenderTarget->GetHeight());
		GfxDevice::SetGlobalTexture(s_PickingTextureId, s_SceneRenderTarget);
		GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), s_ObjectPickerOutlineMaterial));
		GfxDevice::SetViewport(0, 0, viewport.x, viewport.y);
	}
}
