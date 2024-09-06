Shader
{
	Properties
	{
		Texture2D _BaseMap = "white"
	}
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		Cull Front

		HLSLBEGIN
		#pragma vertex IconVertex
		#pragma fragment IconFragment

		#include "Input.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float3 texcoordAlpha : TEXCOORD0;
		};

		float GetAlpha(float dist)
		{
			float fadeoutRange = 2.0f;
			float fadeoutEnd = 1.2f;
			return saturate(pow(dist - fadeoutEnd, 3) / pow(fadeoutRange, 3));
		}

		Varyings IconVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(mul(float4(input.positionOS, 1.0f), _ModelMatrix), _ViewProjectionMatrix);
			output.texcoordAlpha.xy = input.texcoord;
			float3 modelPositionWS = float3(_ModelMatrix._41, _ModelMatrix._42, _ModelMatrix._43);
			output.texcoordAlpha.z = GetAlpha(distance(modelPositionWS, _CameraPositionWS));
			return output;
		}

		Texture2D _BaseMap;
		SamplerState _BaseMap_Sampler;

		float4 IconFragment(Varyings input) : SV_TARGET
		{
			float4 color = _BaseMap.Sample(_BaseMap_Sampler, input.texcoordAlpha.xy);
			color.a *= input.texcoordAlpha.z;
			return color;
		}
		HLSLEND
	}
}