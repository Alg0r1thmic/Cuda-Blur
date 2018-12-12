#include <iostream>
#include <cstdlib>
#include <functional>
#include <map>
#include "kernels.h"
////////////////////////////////
#include "lodepng.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;
//funciones  auxiliares
float* gaussianDistance(float sigma, const int fsize); 
float* gaussianRange(float sigma, const int range) ;
//////////////////////////////////////////////////////////////
//mostrar imagen
#include "CImg.h"
using namespace cimg_library;

int main() {
   
    //Archivos, de entrada
    const char *arch_entrada = "musica.png";
    const char *arch_salida = "musica_blur.png";

    vector<unsigned char> imagen_vector;
    unsigned int width, height;

    // Usamos librería LODEPNG y su manejo de errores
    int error = lodepng::decode(imagen_vector, width, height, arch_entrada);
    if(error)
      cout << "Error de LODEPNG, " << error << ": " << lodepng_error_text(error) << "\n";
        //Para convertir de RGBA a RGB solo hay que dejar de lado la sección alpha
    //Nuestro proceso será sin el canal A de RGBA, por lo tanto el tamaño será menor
    //  por eso se multiplica por 3/4
    unsigned char* imagen_ini = new unsigned char[(imagen_vector.size()*3)/4]; 
    unsigned char* imagen_fin = new unsigned char[(imagen_vector.size()*3)/4];
    int caracter = 0;
    for(int i = 0; i < imagen_vector.size(); i++) {
       if((i+1) % 4 != 0) {
           imagen_ini[caracter] = imagen_vector.at(i);
           imagen_fin[caracter] = 255;
           caracter++;
       }
    }
        
    //CUDA
    blurr(imagen_ini, imagen_fin, width, height); 

    //Salida final
    std::vector<unsigned char> salida_final;
    for(int i = 0; i < imagen_vector.size(); ++i) {
        salida_final.push_back(imagen_fin[i]);
        if((i+1) % 3 == 0) {
            salida_final.push_back(255);
        }
    }
    
    // Guardar datos
    error = lodepng::encode(arch_salida, salida_final, width, height);

    if(error)
      cout << "Error de encoder" << error << ": "<< lodepng_error_text(error) << "\n";



	CImg<unsigned char> salida(arch_salida);
	salida.resize(600,800);
	CImgDisplay disp1(salida, arch_salida);
	//disp1.resize(salida,1);
	CImg<unsigned char> entrada(arch_entrada);
	entrada.resize(600,800);
	CImgDisplay disp2(entrada, arch_entrada);
	//disp2.resize(entrada,1);


	//start event loop
	while(true) {
	     //All the interactive code is inside the event loop
	     cimg_library::CImgDisplay::wait(disp2,disp1);
	}

    delete[] imagen_ini;
    delete[] imagen_fin;
    return 0;

}


float* gaussianDistance(float sigma, const int fsize) {
    const int size = 2*fsize+1;
    float* kernel = new float[size*size]; 
    const float pi = std::atan(1.0f)*4.0f;
    float sigmasquared2 = 2*sigma*sigma;

    for(int x = -fsize; x < fsize+1; ++x) {
        for(int y = -fsize; y < fsize+1; ++y) {
            // (0,0) is center
            float f = expf(-(x*x/sigmasquared2 + y*y/sigmasquared2));
            kernel[x+fsize+(y+fsize)*size] = f / (sigmasquared2*pi);
        }
    }
    
    return kernel;
}
float* gaussianRange(float sigma, const int range) {
    float* kernel = new float[range];
    const float sqrt2pi = 2.0f*sqrt(std::atan(1.0f)*4.0f);
    float sigmasquared2 = 2*sigma*sigma;

    for(int x = 0; x < range; ++x) {
        float f = expf(-(x*x/sigmasquared2));
        kernel[x] = f / (sigma*sqrt2pi);
    }
    return kernel;
}
////////////////
