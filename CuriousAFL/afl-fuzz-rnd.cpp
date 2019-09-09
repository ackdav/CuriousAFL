//
//
//

extern "C" void f(int);
int vote_seed(const seed){
    return 1
}

/*
#include "afl-fuzz-rnd.h"
#include <torch/torch.h>

struct NN  : torch::nn::Module {
    NN() {
        // construct and register your layers
        in = register_module("in",torch::nn::Linear(8,64));
        h = register_module("h",torch::nn::Linear(64,64));
        out = register_module("out",torch::nn::Linear(64,1));
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
    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};
};

class RND {
    NN target
    NN model
    torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(1e-3));

    int vote_seed(const seed){
        return 1
    }
    //auto out = model.forward(in);
};

extern "C" int call_C_veto(RND* p, string seed){
    return p->vote_seed(i);
}
*/

/* example wrapper for functions
extern "C" double call_C_f(C* p, int i) // wrapper function
{
    return p->f(i);
}*/