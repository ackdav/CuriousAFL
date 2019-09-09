//
// Created by david on 09.09.19.
//https://stackoverflow.com/questions/2203023/how-to-call-a-c-class-and-its-method-from-a-c-file
//

#ifndef CURIOUSAFL_AFL_FUZZ_RND_H
#define CURIOUSAFL_AFL_FUZZ_RND_H


#ifdef __cplusplus // only actually define the class if this is C++

class RNDBase
{
    public:
        RND();
        void init_model();
        int veto_seed(string);
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
EXPORT_C void RND_init_model(RND*);
EXPORT_C int RND_veto_seed(RND*, std::string);

#endif //CURIOUSAFL_AFL_FUZZ_RND_H
