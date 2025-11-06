#ifndef MACROS_INCLUDED
#define MACROS_INCLUDED

#define PI 3.14159265358979323846

static uint _ViewIndex;
static uint _RenderInstanceId;

#if (MULTIVIEW)

#define VIEW_INDEX							_ViewIndex
#define VERTEX_INPUT_INSTANCE_ID			uint instanceID : SV_InstanceID; \
											uint renderInstanceID : RENDER_INSTANCE;
#define VERTEX_OUTPUT_VIEW_INDEX			uint renderTargetIndex : SV_RenderTargetArrayIndex;
#define SETUP_INSTANCE_ID(input)			_ViewIndex = input.instanceID % _ViewCount.x; _RenderInstanceId = input.renderInstanceID - (input.instanceID / _ViewCount.x + _ViewIndex)
#define SETUP_OUTPUT_VIEW_INDEX(output)		output.renderTargetIndex = _ViewIndex
#define SETUP_INPUT_VIEW_INDEX(input)		_ViewIndex = input.renderTargetIndex

#else

#define VIEW_INDEX							0
#define VERTEX_INPUT_INSTANCE_ID			uint renderInstanceID : RENDER_INSTANCE;
#define VERTEX_OUTPUT_VIEW_INDEX
#define SETUP_INSTANCE_ID(input)			_RenderInstanceId = input.renderInstanceID
#define SETUP_OUTPUT_VIEW_INDEX(output)
#define SETUP_INPUT_VIEW_INDEX(input)

#endif

#define OBJECT_TO_WORLD_MATRIX				_PerDrawData[_RenderInstanceId].modelMatrix
#define LIGHTMAP_CHART_OFFSET				_PerDrawData[_RenderInstanceId].lightmapChartOffset
#define VIEW_MATRIX							_ViewMatrix[_ViewIndex]
#define VIEW_PROJECTION_MATRIX				_ViewProjectionMatrix[_ViewIndex]
#define INVERSE_VIEW_PROJECTION_MATRIX		_InverseViewProjectionMatrix[_ViewIndex]
#define INVERSE_PROJECTION_MATRIX			_InverseProjectionMatrix[_ViewIndex]
#define CAMERA_POSITION_WS					_CameraPositionWS
#define CAMERA_FORWARD_DIRECTION_WS			_CameraForwardDirectionWS
#define CAMERA_SIZE_INV_SIZE				_CameraSizeInvSize
#define RENDER_TARGET_SIZE_INV_SIZE			_RenderTargetSizeInvSize


#define TEXTURE2D(textureName)							Texture2D textureName
#define TEXTURE2D_UINT(textureName)						Texture2D<uint> textureName
#define TEXTURE2D_MSAA(textureName, samples)			Texture2DMS<float4, samples> textureName
#define TEXTURE2D_MSAA_FLOAT(textureName, samples)		Texture2DMS<float, samples> textureName
#define TEXTURE2D_ARRAY(textureName)					Texture2DArray textureName
#define TEXTURE2D_ARRAY_MSAA(textureName, samples)		Texture2DMSArray<float4, samples> textureName
#define TEXTURE2D_ARRAY_MSAA_FLOAT(textureName, samples)Texture2DMSArray<float, samples> textureName
#define TEXTURECUBE(textureName)						TextureCube textureName
#define TEXTURE3D(textureName)							Texture3D textureName

#define SAMPLER(samplerName)							SamplerState samplerName
#define SAMPLER_CMP(samplerName)						SamplerComparisonState samplerName

#define SAMPLE_TEXTURE2D(textureName, samplerName, coord2)					textureName.Sample(samplerName, coord2)
#define SAMPLE_TEXTURE2D_ARRAY(textureName, samplerName, coord2, index)		textureName.Sample(samplerName, float3(coord2, index))
#define SAMPLE_TEXTURE2D_LOD(textureName, samplerName, coord2, lod)			textureName.SampleLevel(samplerName, coord2, lod)
#define SAMPLE_TEXTURE2D_SHADOW(textureName, samplerName, coord2, depth)    textureName.SampleCmpLevelZero(samplerName, coord2, depth)
#define LOAD_TEXTURE2D(textureName, coord2)									textureName.Load(uint3(coord2, 0))
#define LOAD_TEXTURE2D_MSAA(textureName, uv, sampleIndex)					textureName.Load(uv, sampleIndex)
#define LOAD_TEXTURE2D_ARRAY_MSAA(textureName, uv, index, sampleIndex)		textureName.Load(uint3(uv, index), sampleIndex)
#define SAMPLE_TEXTURECUBE(textureName, samplerName, coord3)				textureName.Sample(samplerName, coord3)
#define SAMPLE_TEXTURECUBE_LOD(textureName, samplerName, coord3, lod)		textureName.SampleLevel(samplerName, coord3, lod)
#define SAMPLE_TEXTURE3D(textureName, samplerName, coord3)					textureName.Sample(samplerName, coord3)
#define SAMPLE_TEXTURE3D_LOD(textureName, samplerName, coord3, lod)			textureName.SampleLevel(samplerName, coord3, lod)

#if (MULTIVIEW)

#define TEXTURE2D_X(textureName)								TEXTURE2D_ARRAY(textureName)
#define TEXTURE2D_X_MSAA(textureName, samples)                  TEXTURE2D_ARRAY_MSAA(textureName, samples)
#define TEXTURE2D_X_MSAA_FLOAT(textureName, samples)            TEXTURE2D_ARRAY_MSAA_FLOAT(textureName, samples)
#define SAMPLE_TEXTURE2D_X(textureName, samplerName, coord2)	SAMPLE_TEXTURE2D_ARRAY(textureName, samplerName, coord2, VIEW_INDEX)
#define LOAD_TEXTURE2D_X_MSAA(textureName, uv, sampleIndex)     LOAD_TEXTURE2D_ARRAY_MSAA(textureName, uv, VIEW_INDEX, sampleIndex)

#else

#define TEXTURE2D_X(textureName)								TEXTURE2D(textureName)
#define TEXTURE2D_X_MSAA(textureName, samples)                  TEXTURE2D_MSAA(textureName, samples)
#define TEXTURE2D_X_MSAA_FLOAT(textureName, samples)            TEXTURE2D_MSAA_FLOAT(textureName, samples)
#define SAMPLE_TEXTURE2D_X(textureName, samplerName, coord2)	SAMPLE_TEXTURE2D(textureName, samplerName, coord2)
#define LOAD_TEXTURE2D_X_MSAA(textureName, uv, sampleIndex)     LOAD_TEXTURE2D_MSAA(textureName, uv, sampleIndex)

#endif

#endif