# good example: https://github.com/apache/thrift/blob/master/tutorial/tutorial.thrift
namespace cpp rnd

service Rnd {
    byte initModel(),
    byte veto(1: string seed)
}
