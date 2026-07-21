// ModelQA PBR Preview helper
// 目的:
//   - 将来のCubemap SRV版PBRで使う反射/屈折関数をまとめる。
//   - 現行Mesh.hlslは手続き環境を直接評価するが、Cubemap化したらここを本流にする。

#include "ModelQAPreviewEnvironment.hlsli"

TextureCube<float4> gPreviewEnvironmentCube : register(t10);
SamplerState gLinearClampSampler : register(s5);

cbuffer IBLConstants : register(b3)
{
    float4 gIBLParams; // x: max mip, y: specular strength, z: diffuse strength, w: reserved
};

float3 MQA_FresnelSchlick(float cosTheta, float3 F0)
{
    const float f = pow(1.0f - saturate(cosTheta), 5.0f);
    return F0 + (1.0f - F0) * f;
}

float3 MQA_SamplePreviewSpecularIBL(float3 N, float3 V, float roughness)
{
    const float3 R = reflect(-V, N);
    const float mip = saturate(roughness) * saturate(roughness) * gIBLParams.x;
    return gPreviewEnvironmentCube.SampleLevel(gLinearClampSampler, R, mip).rgb;
}

float3 MQA_SamplePreviewDiffuseIBL(float3 N)
{
    const float mip = max(gIBLParams.x - 1.5f, 0.0f);
    return gPreviewEnvironmentCube.SampleLevel(gLinearClampSampler, N, mip).rgb;
}

float3 MQA_ApplyPreviewIBL(float3 baseColor, float metallic, float roughness, float3 N, float3 V)
{
    N = MQA_SafeNormalize(N, float3(0.0f, 1.0f, 0.0f));
    V = MQA_SafeNormalize(V, float3(0.0f, 0.0f, 1.0f));

    const float NdotV = saturate(dot(N, V));
    const float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), baseColor, saturate(metallic));
    const float3 F = MQA_FresnelSchlick(NdotV, F0);

    const float3 specularEnv = MQA_SamplePreviewSpecularIBL(N, V, roughness);
    const float3 diffuseEnv  = MQA_SamplePreviewDiffuseIBL(N);

    const float3 diffuse = diffuseEnv * baseColor * (1.0f - metallic);
    const float3 specular = specularEnv * F;
    return diffuse * gIBLParams.z + specular * gIBLParams.y;
}

float3 MQA_ApplyPreviewGlass(float3 glassTint, float roughness, float ior, float3 N, float3 V, float alpha)
{
    N = MQA_SafeNormalize(N, float3(0.0f, 1.0f, 0.0f));
    V = MQA_SafeNormalize(V, float3(0.0f, 0.0f, 1.0f));

    const float NdotV = saturate(dot(N, V));
    const float eta = 1.0f / max(ior, 1.01f);
    const float3 R = reflect(-V, N);
    float3 T = refract(-V, N, eta);
    T = (dot(T, T) > 1.0e-6f) ? MQA_SafeNormalize(T, R) : R;

    const float mip = saturate(roughness) * saturate(roughness) * gIBLParams.x;
    const float3 reflected = gPreviewEnvironmentCube.SampleLevel(gLinearClampSampler, R, mip).rgb;
    const float3 refracted = gPreviewEnvironmentCube.SampleLevel(gLinearClampSampler, T, mip + 0.75f).rgb;

    const float f0 = pow((ior - 1.0f) / (ior + 1.0f), 2.0f);
    const float fresnel = f0 + (1.0f - f0) * pow(1.0f - NdotV, 5.0f);
    const float3 color = lerp(refracted * glassTint, reflected, fresnel);
    return lerp(color, reflected, saturate(1.0f - alpha) * 0.25f);
}
