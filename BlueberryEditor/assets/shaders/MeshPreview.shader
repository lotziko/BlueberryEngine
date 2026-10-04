Shader
{
	Properties
	{
	}
	Pass
	{
		Name "Forward"
		Blend One Zero
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex MeshPreviewVertex
		#pragma fragment MeshPreviewFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float3 normalOS	: NORMAL;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float3 normalWS : TEXCOORD0;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		Varyings MeshPreviewVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = TransformObjectToClip(input.positionOS);
			output.normalWS = TransformObjectToWorldNormal(input.normalOS);

			return output;
		}

		float4 MeshPreviewFragment(Varyings input) : SV_TARGET
		{
			float lambert = dot(input.normalWS, float3(-1, 0, 0));
			return float4(float3(0.5f, 0.5f, 0.5f) * lerp(0.5, 1, lambert), 1.0);
		}
		HLSLEND
	}
	Pass
	{
		Name "Deferred"

		Blend One Zero
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex MeshPreviewVertex
		#pragma fragment MeshPreviewFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float3 normalOS	: NORMAL;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float3 normalWS : TEXCOORD0;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		Varyings MeshPreviewVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = TransformObjectToClip(input.positionOS);
			output.normalWS = TransformObjectToWorldNormal(input.normalOS);

			return output;
		}

		inline void InitializeGBufferData(Varyings input, out GBufferData outGBufferData)
		{
			outGBufferData = (GBufferData)0;
			float lambert = dot(input.normalWS, float3(-1, 0, 0));
			outGBufferData.color.rgb = float3(0.5f, 0.5f, 0.5f) * lerp(0.5, 1, lambert);
			outGBufferData.normalWS.xy = EncodeNormalOctahedral(input.normalWS);
			outGBufferData.orm.rgb = float3(1, 1, 0);
			outGBufferData.bakedGI.rgb = float3(1, 1, 1);
		}
		
		GBufferData MeshPreviewFragment(Varyings input) : SV_TARGET
		{
			SETUP_INPUT_VIEW_INDEX(input);

			GBufferData gBufferData;
			InitializeGBufferData(input, gBufferData);
			return gBufferData;
		}
		HLSLEND
	}
}