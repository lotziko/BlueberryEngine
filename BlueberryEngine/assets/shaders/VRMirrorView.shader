Shader
{
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex VRMirrorViewVertex
		#pragma fragment VRMirrorViewFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
		};

		Varyings VRMirrorViewVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		TEXTURE2D_ARRAY(_ScreenColorTexture);	SAMPLER(_ScreenColorTexture_Sampler);

		float4 VRMirrorViewFragment(Varyings input) : SV_TARGET
		{
			return SAMPLE_TEXTURE2D_ARRAY(_ScreenColorTexture, _ScreenColorTexture_Sampler, input.texcoord, 0);
		}
		HLSLEND
	}
}