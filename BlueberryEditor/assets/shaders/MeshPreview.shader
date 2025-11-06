Shader
{
	Properties
	{
	}
	Pass
	{
		Blend One Zero
		ZWrite On
		Cull Back

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
}