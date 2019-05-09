/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define maxSteps 500
#define tstep 0.01f
__constant float3 light_ambient = (float3)(0.1f, 0.1f, 0.1f);
__constant float3 light_diffuse = (float3)(0.5f, 0.5f, 0.5f);
__constant float3 light_specular = (float3)(0.6f, 0.6f, 0.6f);

__constant float3 material_ambient = (float3)(1.0f, 0.7f, 0.5f);
__constant float3 material_diffuse = (float3)(1.0f, 0.7f, 0.5f);
__constant float3 material_specular = (float3)(0.5f, 0.5f, 0.5f);
// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

int intersectBox(float4 r_o, float4 r_d, float4 boxmin, float4 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float4 invR = (float4)(1.0f,1.0f,1.0f,1.0f) / r_d;
    float4 tbot = invR * (boxmin - r_o);
    float4 ttop = invR * (boxmax - r_o);

    // re-order intersections to find smallest and largest on each axis
    float4 tmin = min(ttop, tbot);
    float4 tmax = max(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

float3 calcNormal(image3d_t volume, sampler_t volumeSampler, float4 pos)
{
	float SobelX[3][3][3] =
	{
		{
			{-1, -3, 1},
			{0, 0, 0},
			{1, 3, 1}
		},
		{
			{-3, -6, -3},
			{0, 0, 0},
			{3, 6, 3}
		},
		{
			{-1, -3, -1},
			{0, 0, 0},
			{1, 3, 1}
		}
	};
	float SobelY[3][3][3] =
	{
		{
			{1, 3, 1},
			{3, 6, 3},
			{1, 3, 1}
		},
		{
			{0, 0, 0},
			{0, 0, 0},
			{0, 0, 0}
		},
		{
			{-1, -3, -1},
			{-3, -6, -3},
			{-1, -3, -1}
		}
	};
	float SobelZ[3][3][3] =
	{
		{
			{-1, 0, 1},
			{-3, 0, 3},
			{-1, 0, 1}
		},
		{
			{-3, 0, 3},
			{-6, 0, 6},
			{-3, 0, 3}
		},
		{
			{-1, 0, 1},
			{-3, 0, 3},
			{-1, 0, 1}
		}
	};
	float4 position = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	float4 ConData[3][3][3];
	float step = 1/306;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				position = (float4)(pos.x + (i - 1) * step, pos.y + (j - 1) * step, pos.z + (k - 1) * step, pos.w);
				//printf("%f %f %f\n",position.x,position.y,position.z);
				//position = (float4)(pos.x, pos.y, pos.z, pos.w);
				ConData[i][j][k] = read_imagef(volume, volumeSampler, position);
				x += ConData[i][j][k].x * SobelX[i][j][k];
				y += ConData[i][j][k].y * SobelY[i][j][k];
				z += ConData[i][j][k].z * SobelZ[i][j][k];
			}
		}
	}
	float3 result = (float3)(x, y, z);
	return result;
}

uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = clamp(rgba.x,0.0f,1.0f);  
    rgba.y = clamp(rgba.y,0.0f,1.0f);  
    rgba.z = clamp(rgba.z,0.0f,1.0f);  
    rgba.w = clamp(rgba.w,0.0f,1.0f);  
    return ((uint)(rgba.w*255.0f)<<24) | ((uint)(rgba.z*255.0f)<<16) | ((uint)(rgba.y*255.0f)<<8) | (uint)(rgba.x*255.0f);
}

__kernel void
d_render(__global uint *d_output, 
         uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale,
         __constant float* invViewMatrix
 #ifdef IMAGE_SUPPORT
          ,__read_only image3d_t volume,
          __read_only image2d_t transferFunc,
          sampler_t volumeSampler,
          sampler_t transferFuncSampler
 #endif
		  ,float len, float wid, float hei
         )

{	

    uint x = get_global_id(0);
    uint y = get_global_id(1);

	float accumA = 0.0;
	float3 accumC = (float3)(0.0,0.0,0.0);

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

	//float tstep = 0.01f;
	float4 boxMin = (float4)(-wid, -hei, -len, 1.0f);
	float4 boxMax = (float4)(wid, hei, len, 1.0f);

    // calculate eye ray in world space
    float4 eyeRay_o;
    float4 eyeRay_d;

    eyeRay_o = (float4)(invViewMatrix[3], invViewMatrix[7], invViewMatrix[11], 1.0f);   

    float4 temp = normalize(((float4)(u, v, -2.0f,0.0f)));
    eyeRay_d.x = dot(temp, ((float4)(invViewMatrix[0],invViewMatrix[1],invViewMatrix[2],invViewMatrix[3])));
    eyeRay_d.y = dot(temp, ((float4)(invViewMatrix[4],invViewMatrix[5],invViewMatrix[6],invViewMatrix[7])));
    eyeRay_d.z = dot(temp, ((float4)(invViewMatrix[8],invViewMatrix[9],invViewMatrix[10],invViewMatrix[11])));
    eyeRay_d.w = 0.0f;

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay_o, eyeRay_d, boxMin, boxMax, &tnear, &tfar);
    if (!hit) {
        if ((x < imageW) && (y < imageH)) {
            // write output color
            uint i =(y * imageW) + x;
            d_output[i] = 0;
        }
        return;
    }
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from back to front, accumulating color
    temp = (float4)(0.0f,0.0f,0.0f,0.0f);
    float t = tfar;

    for(uint i=0; i<maxSteps; i++) {		
        float4 pos = eyeRay_o + eyeRay_d*t;
		pos.x = (pos.x + len) * (0.5/len); 
		pos.y = (pos.y + wid) * (0.5/wid);
		pos.z = (pos.z + hei) * (0.5/hei);

        // read from 3D texture        
#ifdef IMAGE_SUPPORT        
        float4 sampler = read_imagef(volume, volumeSampler, pos);
        // lookup in transfer function texture
        float2 transfer_pos = (float2)((sampler.x-transferOffset)*transferScale, 0.5f);
        float4 col = read_imagef(transferFunc, transferFuncSampler, transfer_pos);
#else
        float4 col = (float4)(pos.x,pos.y,pos.z,.25f);
#endif


        // accumulate result
        float alpha = col.w*density;
		float3 L = eyeRay_d.yxz;
		float3 H = normalize(L + L);
		float3 N = calcNormal(volume, volumeSampler, pos);
		float3 color = (float3)(light_ambient * material_ambient+
									light_diffuse * material_diffuse * max( dot( N, L), 0.0f)+
									light_specular * material_specular * pow( max( dot( N, H), 0.0f), 5));
		accumC = alpha * color + (1-alpha)*accumC;		
		accumA = (1 - accumA) * alpha + accumA;
		col.xyz = accumC;
		col.w = accumA;
        temp = mix(temp, col, (float4)(alpha, alpha, alpha, alpha));
        t -= tstep;
        if (t < tnear) break;
    }
    temp *= brightness;
    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i =(y * imageW) + x;
        d_output[i] = rgbaFloatToInt(temp);
    }
}

