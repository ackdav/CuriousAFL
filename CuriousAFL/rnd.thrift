# good example: https://github.com/apache/thrift/blob/master/tutorial/tutorial.thrift
namespace cpp rnd

service RndService {
    byte initModel(),
    byte veto(1: string seed)
}
