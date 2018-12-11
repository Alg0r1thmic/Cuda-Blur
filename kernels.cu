#include "kernels.h"
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

void getError(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cout << "Error " << cudaGetErrorString(err) << std::endl;
    }
}

__global__
void blur(unsigned char* imagen_ini, unsigned char* imagen_fin, int width, int height) {

    const unsigned int ajuste = blockIdx.x*blockDim.x + threadIdx.x;
    int x = ajuste % width; //situarnos en posición correcta
    int y = (ajuste-x)/width;
    int tam_filtro = 5;
    if(ajuste < width*height) {

        float salida_ch_rojo = 0;
        float salida_ch_verde = 0;
        float salida_ch_azul = 0;
        int cantidad = 0;
        for(int ox = -tam_filtro; ox < tam_filtro+1; ++ox) {
            for(int oy = -tam_filtro; oy < tam_filtro+1; ++oy) {
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    const int ajuste_actual = (ajuste+ox+oy*width)*3;
                    salida_ch_rojo += imagen_ini[ajuste_actual]; 
                    salida_ch_verde += imagen_ini[ajuste_actual+1];
                    salida_ch_azul += imagen_ini[ajuste_actual+2];
                    cantidad++;
                }
            }
        }
        imagen_fin[ajuste*3] = salida_ch_rojo/cantidad;
        imagen_fin[ajuste*3+1] = salida_ch_verde/cantidad;
        imagen_fin[ajuste*3+2] = salida_ch_azul/cantidad;
        }
}

void blurr (unsigned char* imagen_ini, unsigned char* imagen_fin, int width, int height) {
    unsigned char* cuda_input;
    unsigned char* cuda_output;
    getError(cudaMalloc( (void**) &cuda_input, width*height*3*sizeof(unsigned char)));
    getError(cudaMemcpy( cuda_input, imagen_ini, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice ));
 
    getError(cudaMalloc( (void**) &cuda_output, width*height*3*sizeof(unsigned char)));

    dim3 dim_bloque(512,1,1);
    dim3 dim_grid((unsigned int) ceil((double)(width*height*3/dim_bloque.x)), 1, 1 );

    blur<<<dim_grid, dim_bloque>>>(cuda_input, cuda_output, width, height); 
    getError(cudaMemcpy(imagen_fin, cuda_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost ));

    getError(cudaFree(cuda_input));
    getError(cudaFree(cuda_output));
}

