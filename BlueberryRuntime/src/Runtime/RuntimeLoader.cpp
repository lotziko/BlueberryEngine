#include "RuntimeLoader.h"

#include "Blueberry\Serialization\Serializer.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Texture3D.h"
#include "Blueberry\Graphics\TextureCubeArray.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\ComputeShader.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\SceneEvents.h"
#include "Blueberry\Scene\LightingSettings.h"

#include "Concrete\DX11\DX11.h"
#include "Concrete\Windows\ComPtr.h"

#include "Blueberry\Graphics\GfxDevice.h"

#include <fstream>

namespace Blueberry
{
	static HMODULE s_GameAssembly = nullptr;

	void ReadAssetResource(std::ifstream& resourcesStream, Object* object, List<uint8_t>& data)
	{
		TypeId type = object->GetType();
		if (type == Texture2D::Type)
		{
			Texture2D* texture = static_cast<Texture2D*>(object);
			size_t size;
			resourcesStream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
			data.resize(size);
			resourcesStream.read(reinterpret_cast<char*>(data.data()), size);
			if (texture->IsReadable())
			{
				texture->SetData(data.data(), size);
				texture->Apply();
			}
			else
			{
				texture->Apply(data.data(), size);
			}
		}
		else if (type == TextureCube::Type)
		{
			TextureCube* texture = static_cast<TextureCube*>(object);
			size_t size;
			resourcesStream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
			data.resize(size);
			resourcesStream.read(reinterpret_cast<char*>(data.data()), size);
			if (texture->IsReadable())
			{
				texture->SetData(data.data(), size);
				texture->Apply();
			}
			else
			{
				texture->Apply(data.data(), size);
			}
		}
		else if (type == Texture3D::Type)
		{
			Texture3D* texture = static_cast<Texture3D*>(object);
			size_t size;
			resourcesStream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
			data.resize(size);
			resourcesStream.read(reinterpret_cast<char*>(data.data()), size);
			if (texture->IsReadable())
			{
				texture->SetData(data.data(), size);
				texture->Apply();
			}
			else
			{
				texture->Apply(data.data(), size);
			}
		}
		else if (type == TextureCubeArray::Type)
		{
			TextureCubeArray* texture = static_cast<TextureCubeArray*>(object);
			size_t size;
			resourcesStream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
			data.resize(size);
			resourcesStream.read(reinterpret_cast<char*>(data.data()), size);
			if (texture->IsReadable())
			{
				texture->SetData(data.data(), size);
				texture->Apply();
			}
			else
			{
				texture->Apply(data.data(), size);
			}
		}
		else if (type == Shader::Type)
		{
			Shader* shader = static_cast<Shader*>(object);
			List<ComPtr<ID3DBlob>> blobs;
			VariantsData variantsData = {};
			uint32_t vertexShaderCount;
			uint32_t geometryShaderCount;
			uint32_t fragmentShaderCount;
			uint32_t blobsCount;

			resourcesStream.read(reinterpret_cast<char*>(&vertexShaderCount), sizeof(uint32_t));
			variantsData.vertexShaderIndices.resize(vertexShaderCount);
			resourcesStream.read(reinterpret_cast<char*>(variantsData.vertexShaderIndices.data()), sizeof(uint32_t) * vertexShaderCount);
			resourcesStream.read(reinterpret_cast<char*>(&geometryShaderCount), sizeof(uint32_t));
			variantsData.geometryShaderIndices.resize(geometryShaderCount);
			resourcesStream.read(reinterpret_cast<char*>(variantsData.geometryShaderIndices.data()), sizeof(uint32_t) * geometryShaderCount);
			resourcesStream.read(reinterpret_cast<char*>(&fragmentShaderCount), sizeof(uint32_t));
			variantsData.fragmentShaderIndices.resize(fragmentShaderCount);
			resourcesStream.read(reinterpret_cast<char*>(variantsData.fragmentShaderIndices.data()), sizeof(uint32_t) * fragmentShaderCount);
			resourcesStream.read(reinterpret_cast<char*>(&blobsCount), sizeof(uint32_t));

			for (uint32_t i = 0; i < blobsCount; ++i)
			{
				size_t size;
				resourcesStream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
				ComPtr<ID3DBlob> blob;
				HRESULT hr = D3DCreateBlob(size, &blob);
				if (SUCCEEDED(hr))
				{
					resourcesStream.read(reinterpret_cast<char*>(blob->GetBufferPointer()), size);
				}
				blobs.push_back(blob);
				variantsData.shaders.push_back(blob.Get());
			}
			shader->Initialize(variantsData);
			for (auto& blob : blobs)
			{
				blob.Reset();
			}
		}
		else if (type == ComputeShader::Type)
		{
			ComputeShader* computeShader = static_cast<ComputeShader*>(object);
			List<ComPtr<ID3DBlob>> blobs;
			List<void*> shaders;
			uint32_t blobCount;
			resourcesStream.read(reinterpret_cast<char*>(&blobCount), sizeof(uint32_t));

			for (uint32_t i = 0; i < blobCount; ++i)
			{
				size_t size;
				resourcesStream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
				ComPtr<ID3DBlob> blob;
				HRESULT hr = D3DCreateBlob(size, &blob);
				if (SUCCEEDED(hr))
				{
					resourcesStream.read(reinterpret_cast<char*>(blob->GetBufferPointer()), size);
				}
				blobs.push_back(blob);
				shaders.push_back(blob.Get());
			}
			computeShader->Initialize(shaders);
			for (auto& blob : blobs)
			{
				blob.Reset();
			}
		}
	}

	void RuntimeLoader::Load(Scene* scene)
	{
		// Assembly
		String dllPath = "GameAssembly.dll";
		using EntryFunc = void(*)();
		s_GameAssembly = LoadLibraryA(dllPath.c_str());
		if (s_GameAssembly != nullptr)
		{
			EntryFunc entryFunc = (EntryFunc)GetProcAddress(s_GameAssembly, "Entry");
			if (entryFunc != nullptr)
			{
				entryFunc();
			}
		}

		// Assets
		Dictionary<Guid, Object*> assets;
		{
			String assetsPath = "Assets";;
			Serializer assetsSerializer = {};
			assetsSerializer.Deserialize(assetsPath, SerializationFlags::RuntimeOnly | SerializationFlags::HasGuids);
			for (auto& pair : assetsSerializer.GetDeserializedObjects())
			{
				Object* object = ObjectDB::GetObject(pair.first);
				if (ObjectDB::HasGuid(pair.first))
				{
					assets.insert_or_assign(ObjectDB::GetGuidFromObjectId(pair.first), object);
				}
				if (object->GetType() == Mesh::Type)
				{
					Mesh* mesh = static_cast<Mesh*>(object);
					mesh->Apply();
				}
			}
		}

		// Resources
		{
			List<uint8_t> buffer;
			String resourcesPath = "Resources";
			std::ifstream resourcesStream(resourcesPath.data(), std::ios::in | std::ifstream::binary);
			if (resourcesStream.is_open())
			{
				while (true)
				{
					Guid guid;
					resourcesStream.read(reinterpret_cast<char*>(&guid), sizeof(Guid));
					if (resourcesStream.eof())
					{
						break;
					}
					auto it = assets.find(guid);
					if (it != assets.end())
					{
						ReadAssetResource(resourcesStream, it->second, buffer);
					}
				}
				resourcesStream.close();
			}
		}

		// Scene
		{
			String scenePath = "Scene";
			Serializer sceneSerializer = {};
			sceneSerializer.Deserialize(scenePath, SerializationFlags::RuntimeOnly);

			for (auto& pair : sceneSerializer.GetDeserializedObjects())
			{
				Object* object = ObjectDB::GetObject(pair.first);
				if (object->GetType() == Entity::Type)
				{
					scene->AddEntity(static_cast<Entity*>(object));
				}
				else if (object->GetType() == LightingSettings::Type)
				{
					LightingSettings* lightingSettings = static_cast<LightingSettings*>(object);

					// Lightmap
					{
						Texture2D* lightmap = lightingSettings->GetLightmap();
						GfxDevice::SetGlobalTexture(TO_HASH("_LightmapTexture"), lightmap->Get());
					}

					// Probe volume
					{
						Texture3D* probeVolume = lightingSettings->GetProbeVolume();
						GfxDevice::SetGlobalTexture(TO_HASH("_ProbeVolumeTexture"), probeVolume->Get());
					}

					// Charts
					{
						List<Vector4>& chartOffsetScale = lightingSettings->GetChartOffsetScale();
						GfxBuffer* scaleOffsetBuffer;
						BufferProperties properties = {};

						properties.elementCount = static_cast<uint32_t>(chartOffsetScale.size());
						properties.elementSize = sizeof(Vector4);
						properties.data = chartOffsetScale.data();
						properties.dataSize = chartOffsetScale.size() * sizeof(Vector4);
						properties.usageFlags = BufferUsageFlags::StructuredBuffer | BufferUsageFlags::ShaderResource;
						
						GfxDevice::CreateBuffer(properties, scaleOffsetBuffer);
						GfxDevice::SetGlobalBuffer(TO_HASH("_PerLightmapInstanceData"), scaleOffsetBuffer);
					}

					// Reflection probes
					{
						TextureCubeArray* reflectionProbes = lightingSettings->GetReflectionProbes();
						GfxDevice::SetGlobalTexture(TO_HASH("_ReflectionTexture"), reflectionProbes->Get());
					}
				}
			}
		}

		SceneEvents::Poll();
	}
}