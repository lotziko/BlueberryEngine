Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex SkyboxVertex
		#pragma fragment SkyboxFragment

		#pragma keyword_global_vertex MULTIVIEW

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float3 texcoord : TEXCOORD0;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		Varyings SkyboxVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = TransformWorldToClip(CAMERA_POSITION_WS + input.positionOS.xyz);
			output.texcoord = input.positionOS.xyz;

			return output;
		}

		TEXTURECUBE(_BaseMap);		SAMPLER(_BaseMap_Sampler);

		float4 SkyboxFragment(Varyings input) : SV_TARGET
		{
			return SAMPLE_TEXTURECUBE(_BaseMap, _BaseMap_Sampler, input.texcoord);
		}
		HLSLEND
	}
}