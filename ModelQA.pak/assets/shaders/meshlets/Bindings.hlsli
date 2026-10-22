#ifndef MODELQA_MESHLET_BINDINGS
#define MODELQA_MESHLET_BINDINGS
// Together with SceneConstants (56 DWORDs) this consumes exactly 64 DWORDs.
// Descriptor indices are uniform for the whole draw; no lane-varying indices.
cbuffer MeshletBindings : register(b1)
{
    uint gVertexDescriptor;
    uint gMeshletDescriptor;
    uint gBaseColorDescriptor;
    uint gNormalDescriptor;
    uint gMetallicRoughnessDescriptor;
    uint gEmissiveDescriptor;
    uint gOcclusionDescriptor;
    uint gSkyboxDescriptor;
};
#endif
