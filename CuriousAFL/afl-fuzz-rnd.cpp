//
//
//
#include "afl-fuzz-rnd.h"
#include <string.h>
#include <deque>
#include <math.h>       /* pow */
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iterator>

//#include <valarray>


int MAX_FILESIZE = pow(2, 10);
int LEARNING_RATE = 1e-4;
int BUFFER_SIZE = pow(2,10);  // how many seeds to keep in memory
int BATCH_SIZE = pow(10, 4);  // update reference model after X executions

int INPUT_DIM = MAX_FILESIZE;  // input dimension of RND
int H_DIM = pow(2,9);
int OUTPUT_DIM = pow(2,6);  // output dimension of RND



std::deque<float> replay_buffer(128); // = (maxlen=int(BUFFER_SIZE/5))
std::deque<float> reward_buffer(128); // = (maxlen=int(BUFFER_SIZE/5))


template<class BidiIter>
BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        BidiIter r = begin;
        std::advance(r, rand()%left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

torch::DeviceType getTorchDeviceType(){
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    return device_type;
}

std::vector<char> ReadAllBytes(char const* filename)
{
  /*  std::ifstream ifs(filename, std::ios::binary|std::ios::ate);
    std::ifstream::pos_type pos = ifs.tellg();


    ifs.seekg(0, std::ios::beg);
    ifs.read(&seed[0], pos);
*/

    std::ifstream input(filename, std::ios::binary);

    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

void handle_error(const char* msg) {
    perror(msg);
    exit(255);
}

NN::NN(int input_size, int h_size, int output_size){
    in = register_module("in", torch::nn::Linear(input_size,h_size));
    h = register_module("h", torch::nn::Linear(h_size,h_size));
    out = register_module("out", torch::nn::Linear(h_size,output_size));
}

std::vector<torch::Tensor> NN::getParameters() {
    return this->parameters();
}

torch::Tensor NN::forward(torch::Tensor X){
    X = torch::relu(in->forward(X));
    X = torch::relu(h->forward(X));
    X = torch::softmax(out->forward(X),1);
    return X;
}

RND::RND() : target_model(INPUT_DIM, H_DIM, OUTPUT_DIM), predictor_model(INPUT_DIM, H_DIM, OUTPUT_DIM)
        {

    this->target_model = NN(INPUT_DIM, H_DIM, OUTPUT_DIM);
    this->predictor_model =  NN(INPUT_DIM, H_DIM, OUTPUT_DIM);

    //torch::optim::Adam optim(this->predictor_model.parameters(), torch::optim::AdamOptions(1e-3));
    torch::optim::Adam optim(this->predictor_model.getParameters(), torch::optim::AdamOptions(1e-3));
    //torch::optim::Adam optim(predictor_model->parameters(), torch::optim::AdamOptions(1e-3));
    //this->predictor_model.parameters();
        }

float RND::getReward(torch::Tensor X) {
    auto out_target = this->target_model.forward(X).detach();
    auto out_predict = this->predictor_model.forward(X);
    torch::Tensor reward = torch::mse_loss(out_target, out_predict.detach());
    return reward.item<float>();
}

void RND::updateModel(torch::Tensor X) {
    X.backward();
    //this->optim.step();
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

float op_normalize (int i) {
    return (float)i/255;
}

//actual function
int RND::veto_seed(char seed_name[])
{
    puts("-------------------------------------------------------------------");
    std::cout << seed_name << std::endl;
    //printf("%6.4lf",byte_array.size());
    puts("-------------------------------------------------------------------");


    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<char> byte_array(MAX_FILESIZE);
    byte_array = ReadAllBytes(seed_name);

    //std::vector<float> newOne = std::vector<char>( byte_array.begin(), byte_array.end() );
    //std::vector<double> doubleVec(intVec.begin(), intVec.end());

    //std::vector<float> bit_array = std::transform (byte_array.begin(), byte_array.end(), byte_array.begin(), op_normalize);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = end - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << "Reading took: " << ms << std::endl;

    torch::Tensor byte_array_train = torch::from_blob(byte_array.data(), {1,MAX_FILESIZE});

    end = std::chrono::high_resolution_clock::now();
    dur = end - begin;
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << "Conversion to torch took: " << ms << std::endl;

    float reward = getReward(byte_array_train);

    end = std::chrono::high_resolution_clock::now();
    dur = end - begin;
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << "getReward took: " << ms << std::endl;

    printf( "REWARD: %6.4lf", reward );
    //float reward_ = reward.toType(torch::FloatType());
    this->replay_buffer.push_front(byte_array_train);
    //this->reward_buffer.push_front(reward_);


    this->step_counter++;
    /*
    if (this->step_counter > 1000){
        //get random elements to train model with
        std::vector<torch::Tensor> rb = std::vector<torch::Tensor>({replay_buffer.begin(), replay_buffer.end()});

        int replay_buffer_length = rb.size();
        int train_on_k = std::min(BATCH_SIZE, replay_buffer_length);

        random_unique(rb.begin(), rb.end(), train_on_k);
        std::vector<torch::Tensor> train_on = std::vector<torch::Tensor>({rb.begin(), rb.end()-(rb.size()-train_on_k)});

        torch::Tensor train_tensor = torch::from_blob(train_on.data(), {1,train_on_k});
        //torch::Tensor train_on = replay_copy[0 train_on_k];
        torch::Tensor reward = this->getReward(train_tensor);
        this->updateModel(reward);

        this->step_counter = 0;
    }*/
    return 0;
}
//wrapper
EXPORT_C int RND_veto_seed(RND* rnd, char seed[])
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


example wrapper for functions
extern "C" double call_C_f(C* p, int i) // wrapper function
{
    return p->f(i);

}*/

