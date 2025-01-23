Shader
{
	Pass
	{
		Blend One Zero
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex MeshObjectPickerVertex
		#pragma fragment MeshObjectPickerFragment

		#include "Core.hlsl"

		cbuffer PerObjectData
		{
			float4 _ObjectId;
		}

		struct Attributes
		{
			float3 positionOS : POSITION;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
		};

		Varyings MeshObjectPickerVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = mul(mul(float4(input.positionOS, 1.0f), _ModelMatrix), VIEW_PROJECTION_MATRIX);
			return output;
		}

		float4 MeshObjectPickerFragment(Varyings input) : SV_TARGET
		{
			return float4(_ObjectId.r, _ObjectId.g, 0, 1);
		}
		HLSLEND
	}
}