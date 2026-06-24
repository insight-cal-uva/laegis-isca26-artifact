#include <stdio.h>
#include "file_op.h"


int main(){
    
    save_log(__FILE__,"normal",NULL, "hello world\n");
    save_log(__FILE__,"normal","16mb", "hello world\n");
    return 0;
}