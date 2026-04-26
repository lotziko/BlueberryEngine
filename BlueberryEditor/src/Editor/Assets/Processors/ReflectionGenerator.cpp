#include "ReflectionGenerator.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Blueberry\Scene\Components\ReflectionProbe.h"
#include "Blueberry\Graphics\Enums.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"
#include "Blueberry\Graphics\Buffers\PerCameraDataConstantBuffer.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxDevice.h"

#include "Editor\Misc\TextureHelper.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Scene\SceneSettings.h"
#include "Editor\Scene\LightingData.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\TextureImporter.h"

#include <directxtex\DirectXTex.h>

namespace Blueberry
{
	GfxTexture* ReflectionGenerator::s_RenderTargetTexture = nullptr;
	GfxTexture* ReflectionGenerator::s_CubeTexture = nullptr;
	Camera* ReflectionGenerator::s_Camera = nullptr;

	#define SIZE 128
	#define SLICE_SIZE SIZE * SIZE * 8

	static Quaternion s_Rotation[6] = {};
	static List<uint8_t> s_SliceData = {};

	void ReflectionGenerator::GenerateReflectionTexture(SkyRenderer* skyRenderer)
	{
		Initialize();
		EditorSceneManager::Save();
		Transform* cameraTransform = s_Camera->GetTransform();
		cameraTransform->SetLocalPosition(Vector3::Zero);
		GfxDevice::SetRenderTarget(s_RenderTargetTexture);
		GfxDevice::SetViewport(0, 0, SIZE, SIZE);
		for (uint32_t i = 0; i < 6; ++i)
		{
			cameraTransform->SetLocalRotation(s_Rotation[i]);
			PerCameraDataConstantBuffer::BindData(s_Camera, s_RenderTargetTexture);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetCube(), skyRenderer->GetMaterial(), 0));
			s_RenderTargetTexture->GetData(s_SliceData.data() + SLICE_SIZE * i);
		}
		s_CubeTexture->SetData(s_SliceData.data(), SLICE_SIZE * 6);
		GfxDevice::SetRenderTarget(nullptr);

		SceneSettings* settings = EditorSceneManager::GetSettings();
		LightingData* lightingData = settings->GetLightingData();
		if (lightingData == nullptr || lightingData->GetState() != ObjectState::Default)
		{
			lightingData = Object::Create<LightingData>();
			settings->SetLightingData(lightingData);
		}
		TextureCube* result = Save(0);
		lightingData->SetSkyReflection(skyRenderer, result);
		AssetDB::SetDirty(lightingData);
		AssetDB::SaveAssets();
	}

	void ReflectionGenerator::GenerateReflectionTexture(ReflectionProbe* reflectionProbe)
	{
		Initialize();
		EditorSceneManager::Save();
		Transform* cameraTransform = s_Camera->GetTransform();
		cameraTransform->SetLocalPosition(reflectionProbe->GetTransform()->GetPosition());
		for (uint32_t i = 0; i < 6; ++i)
		{
			cameraTransform->SetLocalRotation(s_Rotation[i]);
			DefaultRenderer::Draw(reflectionProbe->GetScene(), s_Camera, Rectangle(0, 0, SIZE, SIZE), s_RenderTargetTexture, nullptr);
			s_RenderTargetTexture->GetData(s_SliceData.data() + SLICE_SIZE * i);
		}
		s_CubeTexture->SetData(s_SliceData.data(), SLICE_SIZE * 6);

		LightingData* lightingData = EditorSceneManager::GetSettings()->GetLightingData();
		uint32_t index = reflectionProbe->GetAtlasIndex();
		if (index == UINT_MAX)
		{
			index = std::max(static_cast<uint32_t>(lightingData->GetReflectionProbeCount()), 1u);
		}
		TextureCube* result = Save(index);
		reflectionProbe->SetAtlasIndex(index);
		lightingData->SetReflectionProbe(index - 1, reflectionProbe, result);
		AssetDB::SetDirty(lightingData);
		AssetDB::SaveAssets();
	}

	void ReflectionGenerator::Initialize()
	{
		if (s_RenderTargetTexture == nullptr)
		{
			TextureProperties textureProperties = {};
			textureProperties.width = SIZE;
			textureProperties.height = SIZE;
			textureProperties.depth = 1;
			textureProperties.antiAliasing = 1;
			textureProperties.mipCount = 1;
			textureProperties.format = TextureFormat::R16G16B16A16_Float;
			textureProperties.dimension = TextureDimension::Texture2D;
			textureProperties.wrapMode = WrapMode::Clamp;
			textureProperties.filterMode = FilterMode::Bilinear;
			textureProperties.usageFlags = TextureUsageFlags::RenderTarget | TextureUsageFlags::CPUReadable;
			GfxDevice::CreateTexture(textureProperties, s_RenderTargetTexture);
			s_RenderTargetTexture->SetName("ReflectionGenerator RenderTarget");

			textureProperties.dimension = TextureDimension::TextureCube;
			textureProperties.usageFlags = TextureUsageFlags::RenderTarget | TextureUsageFlags::CPUWritable;
			GfxDevice::CreateTexture(textureProperties, s_CubeTexture);
			s_CubeTexture->SetName("ReflectionGenerator CubeTexture");
		}

		if (s_Camera == nullptr)
		{
			Entity* cameraEntity = Object::Create<Entity>();
			cameraEntity->AddComponent<Transform>();
			s_Camera = cameraEntity->AddComponent<Camera>();
			s_Camera->SetCameraType(CameraType::Reflection);
			s_Camera->SetOrthographic(false);
			s_Camera->SetFieldOfView(90.0f);
			s_Camera->SetAspectRatio(1.0f);
			cameraEntity->OnCreate();

			s_Rotation[0] = Quaternion::CreateFromAxisAngle(Vector3::Up, Math::Math::ToRadians(90));
			s_Rotation[1] = Quaternion::CreateFromAxisAngle(Vector3::Up, Math::Math::ToRadians(-90));
			s_Rotation[2] = Quaternion::CreateFromAxisAngle(Vector3::Right, Math::Math::ToRadians(-90));
			s_Rotation[3] = Quaternion::CreateFromAxisAngle(Vector3::Right, Math::Math::ToRadians(90));
			s_Rotation[4] = Quaternion::Identity;
			s_Rotation[5] = Quaternion::CreateFromAxisAngle(Vector3::Up, Math::Math::ToRadians(180));

			s_SliceData.resize(SLICE_SIZE * 6);
		}
	}

	TextureCube* ReflectionGenerator::Save(const uint32_t& index)
	{
		DirectX::ScratchImage image = {};
		image.InitializeCube(DXGI_FORMAT_R16G16B16A16_FLOAT, SIZE, SIZE, 1, 1);
		TextureHelper::DownscaleTextureCube(s_CubeTexture, image);

		String path = EditorSceneManager::GetPath();
		String name = "\\ReflectionTexture";
		name.append(std::to_string(index));
		name.append(".hdr");
		path.replace(path.find(".scene"), 6, name);

		std::filesystem::create_directories(std::filesystem::path(path).parent_path());
		HRESULT hr = DirectX::SaveToHDRFile(*image.GetImages(), WString(path.begin(), path.end()).c_str());
		if (FAILED(hr))
		{
			BB_ERROR("Failed to save texture.");
		}

		AssetDB::Refresh();
		auto relativePath = std::filesystem::relative(path, Path::GetAssetsPath());
		TextureImporter* importer = static_cast<TextureImporter*>(AssetDB::GetImporter(StringHelper::ToString(relativePath)));
		importer->SetReadable(true);
		importer->SetTextureShape(TextureImporter::TextureShape::TextureCube);
		importer->SetTextureFormat(TextureImporter::TextureFormat::BC6H);
		importer->SetTextureCubeType(TextureImporter::TextureCubeType::Slices);
		importer->SetTextureCubeIBLType(TextureImporter::TextureCubeIBLType::Specular);
		importer->SaveAndReimport();
		Object* obj = ObjectDB::GetObjectFromGuid(importer->GetGuid(), importer->GetMainObject());
		return static_cast<TextureCube*>(obj);
	}
}
