# good example: https://github.com/apache/thrift/blob/master/tutorial/tutorial.thrift
namespace cpp rnd

service RndService {
    double initModel(),
    # byte veto(1: string seed, 2: i32 len, 3: string out_file)
    double veto(1: string seed)
}