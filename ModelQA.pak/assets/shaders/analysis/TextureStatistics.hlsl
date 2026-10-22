struct Partial {
    uint4 sums;
    uint4 minimum;
    uint4 maximum;
    uint4 counts; // pixels, alpha zero, alpha opaque, alpha intermediate
    uint groupIndex;
    uint invalid;
};
StructuredBuffer<uint> Pixels : register(t0);
RWStructuredBuffer<Partial> Partials : register(u0);
cbuffer Parameters : register(b0) { uint PixelCount; uint GroupCount; };
groupshared uint4 Sum[256];
groupshared uint4 Minimum[256];
groupshared uint4 Maximum[256];
groupshared uint4 Counts[256];
[numthreads(256,1,1)]
void CSMain(uint3 group : SV_GroupID,uint lane : SV_GroupIndex) {
    uint inputs,stride,outputs,outputStride;
    Pixels.GetDimensions(inputs,stride);Partials.GetDimensions(outputs,outputStride);
    // Uniform return before any barrier; malformed output views never write.
    if(group.x>=GroupCount || group.x>=outputs || outputStride!=72)return;
    bool invalid=stride!=4 || inputs<PixelCount || PixelCount>65536 || GroupCount!=(PixelCount+255)/256;
    uint pixel=group.x*256+lane;
    uint4 rgba=0;bool valid=!invalid && pixel<PixelCount;
    if(valid) {uint packed=Pixels[pixel];rgba=uint4(packed&255,(packed>>8)&255,(packed>>16)&255,packed>>24);}
    Sum[lane]=rgba;Minimum[lane]=valid?rgba:uint4(255,255,255,255);Maximum[lane]=rgba;
    Counts[lane]=uint4(valid?1:0,valid&&rgba.w==0?1:0,valid&&rgba.w==255?1:0,valid&&rgba.w>0&&rgba.w<255?1:0);
    GroupMemoryBarrierWithGroupSync();
    for(uint step=128;step>0;step>>=1) {
        if(lane<step) {Sum[lane]+=Sum[lane+step];Minimum[lane]=min(Minimum[lane],Minimum[lane+step]);
            Maximum[lane]=max(Maximum[lane],Maximum[lane+step]);Counts[lane]+=Counts[lane+step];}
        GroupMemoryBarrierWithGroupSync();
    }
    if(lane==0) {Partial p;p.sums=Sum[0];p.minimum=Minimum[0];p.maximum=Maximum[0];p.counts=Counts[0];
        p.groupIndex=group.x;p.invalid=invalid?1:0;Partials[group.x]=p;}
}
