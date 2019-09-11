//
//
//
#include "afl-fuzz-rnd.h"
#include <string.h>
#include <deque>
#include <math.h>       /* pow */
#include <torch/torch.h>


int MAX_FILESIZE = pow(2, 12);
int LEARNING_RATE = 1e-4;
int BUFFER_SIZE = pow(2,10);  // how many seeds to keep in memory
int BATCH_SIZE = pow(10, 4);  // update reference model after X executions

int INPUT_DIM = MAX_FILESIZE;  // input dimension of RND
int H_DIM = pow(2,9);
int OUTPUT_DIM = pow(2,6);  // output dimension of RND



std::deque<float> replay_buffer; // = (maxlen=int(BUFFER_SIZE/5))
std::deque<float> reward_buffer; // = (maxlen=int(BUFFER_SIZE/5))

NN::NN(int input_size, int h_size, int output_size){
    in = register_module("in", torch::nn::Linear(input_size,h_size));
    h = register_module("h", torch::nn::Linear(h_size,h_size));
    out = register_module("out", torch::nn::Linear(h_size,output_size));
}

torch::Tensor NN::forward(torch::Tensor X){
    // let's pass relu
    X = torch::relu(in->forward(X));
    X = torch::relu(h->forward(X));
    X = torch::sigmoid(out->forward(X));

    // return the output
    return X;
}

RND::RND() : target_model(INPUT_DIM, H_DIM, OUTPUT_DIM), predictor_model(INPUT_DIM, H_DIM, OUTPUT_DIM) {
    this->target_model = NN(INPUT_DIM, H_DIM, OUTPUT_DIM);
    this->predictor_model =  NN(INPUT_DIM, H_DIM, OUTPUT_DIM);
}

float RND::getReward(torch::Tensor X) {

}

// access functions
EXPORT_C RND* RND_new(void)
{
    return new RND();
}


EXPORT_C void RND_delete(RND* rnd)
{
    delete rnd;
}


//actual function
int RND::veto_seed(int seed)
{
    return 0;
}
//wrapper
EXPORT_C int RND_veto_seed(RND* rnd, int seed)
{
    return rnd->veto_seed(seed);
}


/*
struct max_deque {
    void push_front(float arr[] c) {
        if (internal_container.size() < MAX_SIZE)
            internal_container.push_front(std::move(c));
        else
            ; // do something else
    }
private:
    int MAX_SIZE = BUFFER_SIZE/5;
    std::deque<float arr[]> internal_container;
}*/

/*
struct NN  : torch::nn::Module {
    NN() {
        // construct and register your layers
   }

    // the forward operation (how data will flow from layer to layer)
    torch::Tensor forward(torch::Tensor X){
        // let's pass relu
        X = torch::relu(in->forward(X));
        X = torch::relu(h->forward(X));
        X = torch::sigmoid(out->forward(X));

        // return the output
        return X;
    }
};*/
/*


class RND {

    //NN target;
    //NN model;
    //
    int veto_seed(char* seed){
        return 1
    }
    void init_model(){

    }
    //auto out = model.forward(in);
};



EXPORT_C int RND_init_model(RND* this)
{
    return this->init_model();
}

EXPORT_C int RND_veto_seed(RND* this, char* seed)
{
    return this->veto_seed(char* seed);
}
/*
class RND {

    //NN target;
    //NN model;
    //torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(1e-3));

    int veto_seed(char* seed){
        return 1
    }
    void init_model(){

    }
    //auto out = model.forward(in);
};

extern "C" int call_C_veto(RND* p, string seed){
    return p->vote_seed(i);
}
*/

/*example wrapper for functions
extern "C" double call_C_f(C* p, int i) // wrapper function
{
    return p->f(i);
}*/

