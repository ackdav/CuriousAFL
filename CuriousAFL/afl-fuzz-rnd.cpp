//
//
//
#include "afl-fuzz-rnd.h"
#include <string.h>
//#include <torch/torch.h>

/*
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
    //torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};
};
*/


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


// access functions
EXPORT_C RND* RND_new(void)
{
    return new RND();
}

/*
EXPORT_C void RND_delete(RND* this)
{
    delete this;
}*/


EXPORT_C int RND::veto_seed(int seed)
{
    return 1;
}

EXPORT_C int RND_veto_seed(RND* rnd, int seed)
{
    return rnd->veto_seed(seed);
}