/*#include <iostream>
#include <stdio.h>
#include <stdlib.h>


#include <Libloaderapi.h>

#include <chrono>


#include "specgen.h"


size_t stftBufFloatCount;
float *stftBuf;
size_t chunkFloatCount;


#include <list>
std::list<float*> chunksList;


extern "C"
{
    double *wavFileInfo(char *filename);
    double *spectrogramFFIConfirmMemorySize(int imgWidth, int imgHeight, int channels);
    double *spectrogramFFIReturnDouble(char *filename, int fftLen, int userStart, int userEnd, double freqStart, double freqEnd, int nThreads, int dbg, int imgWidth, int imgHeight, int channels, int rulerOnly);
    double *waveformFFIConfirmMemorySize(int userStart, int userEnd, int channel, int canvasWidth);
    double *getWaveformFFIReturnDouble(char *filename, int userStart, int userEnd, int canvasWidth);
    unsigned int getTimeFrames(float fs, int fftLen, int userStart, int userEnd, int imgWidth);
    float *getAxis(float fs, int fftLen, int userStart, int userEnd, double freqStart, double freqEnd, char dbg, int imgWidth, int imgHeight, int totalFrames);
    float *getSpecPixel(char *filename, int fftLen, int userStart, int userEnd, double freqStart, double freqEnd, int nThreads, char dbg, int imgWidth, int imgHeight, int channels);
}


SpecGen::SpecGen() {
};

void getFileInfo(sapi_context_t* context, sapi_ipc_message_t* message, const sapi_ipc_router_t* router){
    sapi_ipc_result_t* result = sapi_ipc_result_create(context, message);
    sapi_context_t* processContext = sapi_context_create(context, true);


    //std::cout << "getting file info..." << std::endl;


    auto filePath = (char *)sapi_ipc_message_get(message, "filePath");

    


    auto res = wavFileInfo(filePath);

    
    //std::cout << "filePath = " << filePath << std::endl;
    //std::cout << "res = " << res << std::endl;

    auto data = sapi_json_object_create(context);

    sapi_json_object_set(
      data,
      "sampleCount",
      sapi_json_any(sapi_json_number_create(context, res[0]))
    );

    sapi_json_object_set(
      data,
      "channelChunt",
      sapi_json_any(sapi_json_number_create(context, res[1]))
    );

    sapi_json_object_set(
      data,
      "sampleRate",
      sapi_json_any(sapi_json_number_create(context, res[2]))
    );

    sapi_json_object_set(
      data,
      "bitDepth",
      sapi_json_any(sapi_json_number_create(context, res[3]))
    );



    sapi_ipc_result_set_json_data(result, sapi_json_any(data));

    sapi_ipc_reply(result);
}




void generateSpectrogram(sapi_context_t* context, sapi_ipc_message_t* message, const sapi_ipc_router_t* router){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    sapi_ipc_result_t* result = sapi_ipc_result_create(context, message);
    sapi_context_t* processContext = sapi_context_create(context, true);


    //std::cout << "generating spectrogram..." << std::endl;


    auto path = (char *)sapi_ipc_message_get(message, "path");
    //std::cout << "path = " << path << std::endl;

    int imgWidth = atoi(sapi_ipc_message_get(message, "imgWidth"));
    //std::cout << "imgWidth = " << imgWidth << std::endl;
    int imgHeight = atoi(sapi_ipc_message_get(message, "imgHeight"));
    //std::cout << "imgHeight = " << imgHeight << std::endl;

    double *__totalDataSize = spectrogramFFIConfirmMemorySize(imgWidth, imgHeight, 2);
    double totalDataSize = *__totalDataSize;
    //std::cout << "totalDataSize = " << totalDataSize << std::endl;

    unsigned int specDataSize = (unsigned int)totalDataSize * 8;
    //std::cout << "specDataSize = " << specDataSize << std::endl;

    int fftLength = atoi(sapi_ipc_message_get(message, "fftLength"));
    //std::cout << "fftLength = " << fftLength << std::endl;
    int sampleStart = atoi(sapi_ipc_message_get(message, "sampleStart"));
    //std::cout << "sampleStart = " << sampleStart << std::endl;
    int sampleEnd = atoi(sapi_ipc_message_get(message, "sampleEnd"));
    //std::cout << "sampleEnd = " << sampleEnd << std::endl;
    double freqLow = atof(sapi_ipc_message_get(message, "freqLow"));
    //std::cout << "freqLow = " << freqLow << std::endl;
    double freqHigh = atof(sapi_ipc_message_get(message, "freqHigh"));
    //std::cout << "freqHigh = " << freqHigh << std::endl;
    double *specData = spectrogramFFIReturnDouble(path, fftLength, sampleStart, sampleEnd, freqLow, freqHigh, 1, 0, imgWidth, imgHeight, 2, 0);


    //std::cout << "specData acquired" << std::endl;

    sapi_ipc_result_set_bytes(result, specDataSize, (unsigned char *)specData);
    sapi_ipc_reply(result);

    
	free(__totalDataSize);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto µs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    //std::cout << "=====    Time difference: " << ms << "ms  " << µs << "µs  " << ns << "ns" << std::endl;
}


void generateWaveform(sapi_context_t* context, sapi_ipc_message_t* message, const sapi_ipc_router_t* router){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    sapi_ipc_result_t* result = sapi_ipc_result_create(context, message);
    sapi_context_t* processContext = sapi_context_create(context, true);


    //std::cout << "generating waveform..." << std::endl;


    auto path = (char *)sapi_ipc_message_get(message, "path");
    //std::cout << "path = " << path << std::endl;

    int imgWidth = atoi(sapi_ipc_message_get(message, "imgWidth"));
    //std::cout << "imgWidth = " << imgWidth << std::endl;
    int sampleStart = atoi(sapi_ipc_message_get(message, "sampleStart"));
    //std::cout << "sampleStart = " << sampleStart << std::endl;
    int sampleEnd = atoi(sapi_ipc_message_get(message, "sampleEnd"));
    //std::cout << "sampleEnd = " << sampleEnd << std::endl;
    

    double *__totalDataSize = waveformFFIConfirmMemorySize(sampleStart, sampleEnd, 2, imgWidth);
    double totalDataSize = *__totalDataSize;
    //std::cout << "totalDataSize = " << totalDataSize << std::endl;

    unsigned int wavDataSize = (unsigned int)totalDataSize * 8;
    //std::cout << "wavDataSize = " << wavDataSize << std::endl;


    double *wavData = getWaveformFFIReturnDouble(path, sampleStart, sampleEnd, imgWidth);


    //std::cout << "wavData acquired" << std::endl;

    sapi_ipc_result_set_bytes(result, wavDataSize, (unsigned char *)wavData);
    sapi_ipc_reply(result);

    
	free(__totalDataSize);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();


    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto µs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    //std::cout << "=====    Time difference: " << ms << "ms  " << µs << "µs  " << ns << "ns" << std::endl;
}



void generateSTFTData(sapi_context_t* context, sapi_ipc_message_t* message, const sapi_ipc_router_t* router){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    sapi_ipc_result_t* result = sapi_ipc_result_create(context, message);
    sapi_context_t* processContext = sapi_context_create(context, true);


    std::cout << "generating raw stft data..." << std::endl;


    auto path = (char *)sapi_ipc_message_get(message, "path");
    //std::cout << "path = " << path << std::endl;

    int imgWidth = atoi(sapi_ipc_message_get(message, "imgWidth"));
    //std::cout << "imgWidth = " << imgWidth << std::endl;    
    int imgHeight = atoi(sapi_ipc_message_get(message, "imgHeight"));
    //std::cout << "imgHeight = " << imgHeight << std::endl;
    int fftLength = atoi(sapi_ipc_message_get(message, "fftLength"));
    //std::cout << "fftLength = " << fftLength << std::endl;
    int sampleStart = atoi(sapi_ipc_message_get(message, "sampleStart"));
    //std::cout << "sampleStart = " << sampleStart << std::endl;
    int sampleEnd = atoi(sapi_ipc_message_get(message, "sampleEnd"));
    //std::cout << "sampleEnd = " << sampleEnd << std::endl;
    double freqLow = atof(sapi_ipc_message_get(message, "freqLow"));
    //std::cout << "freqLow = " << freqLow << std::endl;
    double freqHigh = atof(sapi_ipc_message_get(message, "freqHigh"));
    //std::cout << "freqHigh = " << freqHigh << std::endl;



	int wavCh = 2;
	int halfLen = fftLength / 2 + 1;
	float fs = 44100.0f;

    unsigned int totalFrames = getTimeFrames(fs, fftLength, sampleStart, sampleEnd, imgWidth);
    //std::cout << "totalFrames = " << totalFrames << std::endl;

	float *axis = getAxis(fs, fftLength, sampleStart, sampleEnd, freqLow, freqHigh, 0, imgWidth, imgHeight, totalFrames);
	
    stftBufFloatCount = halfLen * totalFrames * wavCh;
    //std::cout << "stftBufFloatCount = " << stftBufFloatCount << std::endl;

    size_t stftBufByteCount = stftBufFloatCount * sizeof(float);
    std::cout << "stftBufByteCount = " << stftBufByteCount << std::endl;


	stftBuf = getSpecPixel(path, fftLength, sampleStart, sampleEnd, 1, 22050, 1, 0, imgWidth, 381, wavCh);
    //std::cout << "stftBuf = " << stftBuf << std::endl;

    size_t sessionID = atoll(sapi_ipc_message_get(message, "sessionID"));
    //std::cout << "sessionID = " << sessionID << std::endl;

    size_t maxChunkSize = atoi(sapi_ipc_message_get(message, "maxChunkSize"));
    //std::cout << "maxChunkSize = " << maxChunkSize << std::endl;

    size_t chunksCount = (size_t)ceil(stftBufByteCount / (double)maxChunkSize);
    //std::cout << "chunksCount = " << chunksCount << std::endl;

    size_t bytesLeft = stftBufByteCount;
    //std::cout << "bytesLeft = " << bytesLeft << std::endl;
    for (int i = 0; i < chunksCount; i++){
        if (bytesLeft == 0) break;

        //std::cout << "-> chunking #" << i << std::endl;

        size_t thisChunkSize = maxChunkSize;
        //std::cout << "    thisChunkSize = " << thisChunkSize << std::endl;

        if (thisChunkSize > bytesLeft) thisChunkSize = bytesLeft;
        //std::cout << "    thisChunkSize = " << thisChunkSize << std::endl;

        char *chunkData = (char *)malloc(thisChunkSize);
        //std::cout << "    chunk address = " << (void*)chunkData << std::endl;

        
        size_t offset = maxChunkSize * i;
        //std::cout << "    offset = " << offset << std::endl;


        //std::cout << "    copying bytes..." << chunkData << std::endl;
        memcpy(chunkData, (char *)stftBuf + offset, thisChunkSize);


    

        /*  !!! IMPORTANT !!!
            PROHIBITED TO USE A RELATIVE PATH e.g "tmp/chunk.dat" WHEN socket.ini
            has [webview] watch = true and [webview.watch] reload = true.
            
            THIS WILL CAUSE UI TO GO BLANK!!!
        */
  /*      char filename[256];
        snprintf(filename, sizeof(filename), "tmp/stft_chunks/%zu_chunk_%d.dat", sessionID, i); 
        
        //std::cout << "    writing bytes to " << filename << std::endl;

        FILE *fpc = fopen(filename, "wb");
        //std::cout << "        opened " << filename << std::endl;
        if (fpc != nullptr) {
            //std::cout << "        writing..." << std::endl;
            fwrite(chunkData, 1, thisChunkSize, fpc);
            //std::cout << "        wrote!" << std::endl;

            //std::cout << "        closing..." << std::endl;
            fclose(fpc);
        } else {
            //std::cerr << "Failed to open file " << filename << std::endl;
        }
        
        //std::cout << "    freeing..." << std::endl;
        //free(chunkData);

        //std::cout << "        Done!" << chunkData << std::endl;

        if (maxChunkSize < bytesLeft) bytesLeft -= maxChunkSize;
        else bytesLeft -= bytesLeft;
        //std::cout << "    bytesLeft = " << bytesLeft << std::endl;
    }


    //std::cout << "freeing stftBuf..." << std::endl;
    free(stftBuf);
    
    //std::cout << "freeing axis..." << std::endl;
	free(axis);





    auto data = sapi_json_object_create(context);

    sapi_json_object_set(data,
        "totalFrames",
        sapi_json_any(sapi_json_number_create(context, totalFrames))
    );

    sapi_json_object_set(data,
        "chunksCount",
        sapi_json_any(sapi_json_number_create(context, chunksCount))
    );

    
    sapi_json_object_set(data,
        "fullDataSize",
        sapi_json_any(sapi_json_number_create(context, stftBufByteCount))
    );

    sapi_ipc_result_set_json_data(result, sapi_json_any(data));

    sapi_ipc_reply(result);
    

    /*
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto µs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    //std::cout << "=====    Time difference: " << ms << "ms  " << µs << "µs  " << ns << "ns" << std::endl;*/
/*}


void getRawSTFTDataChunk(sapi_context_t* context, sapi_ipc_message_t* message, const sapi_ipc_router_t* router){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    sapi_ipc_result_t* result = sapi_ipc_result_create(context, message);
    sapi_context_t* processContext = sapi_context_create(context, true);



    int chunksCount = atoi(sapi_ipc_message_get(message, "chunksCount"));
    //std::cout << "chunksCount = " << chunksCount << std::endl;    
    int chunkIdx = atoi(sapi_ipc_message_get(message, "chunkIdx"));
    //std::cout << "chunkIdx = " << chunkIdx << std::endl;

    
    std::cout << "getting raw stft data for chunk " << chunkIdx << "..." << std::endl;



    int idx = -1;
    for (float* chunkPtr : chunksList) {
        idx++;
        if (idx == chunkIdx){
            std::cout << chunkPtr << std::endl; // Print the value pointed to by each element
            sapi_ipc_result_set_bytes(result, (unsigned int)chunkFloatCount * sizeof(float), (unsigned char *)chunkPtr);
            break;
        }
    }

    
    //std::cout << "6" << std::endl;
    sapi_ipc_reply(result);
    //std::cout << "7" << std::endl;

    
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto µs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << "=====    Time for chunk " << chunkIdx << ": " << ms << "ms  " << µs << "µs  " << ns << "ns" << std::endl;
    
}

/************/

/*bool initialize(sapi_context_t* context, const void* data) {
    if (sapi_extension_is_allowed(context, "ipc,ipc_router,ipc_router_map")) {
        //sapi_ipc_router_map(context, "specgen.process.spawn", onProcSpawn, data);
        //sapi_ipc_router_map(context, "specgen.testFunc", testFunc, data);

        sapi_ipc_router_map(context, "specgen.getFileInfo"          , getFileInfo           , data);
        sapi_ipc_router_map(context, "specgen.generateSpectrogram"  , generateSpectrogram   , data);
        sapi_ipc_router_map(context, "specgen.generateWaveform"     , generateWaveform      , data);
        sapi_ipc_router_map(context, "specgen.generateSTFTData"     , generateSTFTData      , data);
        sapi_ipc_router_map(context, "specgen.getRawSTFTDataChunk"  , getRawSTFTDataChunk   , data);
    }
    


    return true;
}

bool deinitialize (sapi_context_t* context, const void *data) {
    return true;
}

SOCKET_RUNTIME_REGISTER_EXTENSION(
    "specgen", // name
    initialize, // initializer
    deinitialize, // deinitializer
    "SpecGen", // description
    "0.1.0" // version
);
*/