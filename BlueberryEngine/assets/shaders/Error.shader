Shader
{
	Pass
	{
		Blend One Zero
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex ErrorVertex
		#pragma fragment ErrorFragment

		#pragma keyword_global_vertex MULTIVIEW

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

		Varyings ErrorVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = TransformObjectToClip(input.positionOS);
			output.normalWS = TransformObjectToWorldNormal(input.normalOS);

			return output;
		}

		float4 ErrorFragment(Varyings input) : SV_TARGET
		{
			float lambert = dot(input.normalWS, -CAMERA_FORWARD_DIRECTION_WS);
			return float4(1.0 * lambert, 0.0, 1.0 * lambert, 1.0);
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZWrite On
		Cull Front

		HLSLBEGIN
		#pragma vertex ErrorVertex
		#pragma fragment ErrorFragment

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
			VERTEX_OUTPUT_VIEW_INDEX
		};

		Varyings ErrorVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = TransformObjectToClip(input.positionOS);
			return output;
		}

		void ErrorFragment(Varyings input)
		{
		}
		HLSLEND
	}
}