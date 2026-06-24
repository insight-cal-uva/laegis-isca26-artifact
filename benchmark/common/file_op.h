#pragma once

#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include<string>

// let the file name be __FILE__
// usage: save_log(__FILE__,"normal",NULL, "hello world %f\n", 12.4);
// usage: save_log(__FILE__,"normal","16mb", "hello world\n");
static int save_log(char *log_file, char *mode, char *size, char* fmt, ...){

    // char temp_name [128];
    // strcpy(temp_name, log_file);
    
    // // remove the postfix
    // int len = strlen(temp_name);
    // for (int i = 0; i < len; i++){
    //     if (temp_name[i] == '.'){
    //         temp_name[i] = '\0';
    //         break;
    //     }
    // }

    // const char * path = "/asplos/uvm_bench/log/";
    
    // char filepath[256];

    
    // if (size == NULL){
    //     // ..../log/normal-gemm.txt
    //     sprintf(filepath, "%s%s-%s.txt", path, mode, temp_name);
    // }else{
    //     // ..../log/normal-gemm-16mb.txt
    //     sprintf(filepath, "%s%s-%s-%s.txt", path, mode, temp_name,size);
    // }

    // printf("file path: %s\n", filepath);

    // FILE *fp = fopen(filepath, "a");
    // if(fp == NULL){
    //     printf("open file failed\n");
    //     exit(0);
    // }
    // va_list args;
    // va_start(args, fmt);
    // vfprintf(fp, fmt, args);
    // va_end(args);
    // fclose(fp);

    return 0;
}


static int save_log_wrapper(std::string filename, float time){
    // std::string save_name = "/asplos/uvm_bench/log/"+filename+".txt";
    // FILE * fp = fopen(save_name.c_str(), "a");
	// if (fp == NULL) {
	// 	fprintf(stderr, "Error opening file!\n");
	// 	exit(1);
	// }
	// fprintf(fp, "%f\n", time);
    // fclose(fp);
    return 0;
}