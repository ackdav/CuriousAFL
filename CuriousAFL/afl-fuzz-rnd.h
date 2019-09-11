//
// Created by david on 09.09.19.
//https://stackoverflow.com/questions/2203023/how-to-call-a-c-class-and-its-method-from-a-c-file
//

#ifndef CURIOUSAFL_AFL_FUZZ_RND_H
#define CURIOUSAFL_AFL_FUZZ_RND_H

#ifdef __cplusplus // only actually define the class if this is C++
#include <torch/torch.h>

class NN;
class RND;

class NN : torch::nn::Module
{
    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};
public:
    NN(int input_size, int h_size, int output_size);
    torch::Tensor forward(torch::Tensor X);
};

class RND
{
    private:

        //torch::optim::Adam optim(predictor_model->parameters(), torch::optim::AdamOptions(1e-3));
        float getReward(torch::Tensor X);
        void updateModel(torch::Tensor X);
public:
        NN target_model;
        NN predictor_model;
        RND();
        void init_model();
        int veto_seed(int);
};

#else
// C doesn't know about classes, just say it's a struct
typedef struct RND RND;
#endif

// access functions
#ifdef __cplusplus
    #define EXPORT_C extern "C"
#else
    #define EXPORT_C
#endif


EXPORT_C RND* RND_new(void);
EXPORT_C void RND_delete(RND*);
//EXPORT_C void RND_init_model(RND*);

EXPORT_C int RND_veto_seed(RND*, int test);

#endif //CURIOUSAFL_AFL_FUZZ_RND_H
