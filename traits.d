/** BLASやLAPACKやベクトル,行列型の特性を調べます
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.traits;

import std.traits;

//public import scid.bindings.blas.types;
import dblet.matrix;
import dblet.vector;

version(unittest){
    import std.stdio;
    import std.typetuple;
    import std.complex;
    
    
    import dblet.lapack.tri;
    import dblet.lapack.trf;
    
    alias TypeTuple!(float, cfloat, double, cdouble) AllBlasType;
    
    pragma(lib, "dblet");
}

unittest{
    auto x = Vector!double([1, 2, 3]),
         y = Vector!double([4, 5, 6]),
         a = Matrix!double(3, 3, [2., 3., 4., 1., 2., 3., 4., 5., 6.]),
         b = Matrix!double(3, 3, [1., 2., 1., 2., 1., 1., 1., 1., 2.]);
    
    auto axpy = 3.0 * x + y;
    writeln(typeof(axpy).stringof);
    writeln(axpy);
    writeln(axpy.evalExpr);
    writeln("");
    
    auto gemv = a * 4.0 * (x * 3.0) + axpy * 2.5;
    writeln(typeof(gemv).stringof);
    writeln(gemv);
    writeln(gemv.evalExpr);
    writeln("");
    
    auto gemm = a * 9.0 * b * 12.0 + (a * 3.0 + a);
    writeln(typeof(gemm).stringof);
    writeln(gemm);
    writeln(gemm.evalExpr);
    writeln("");
    writeln(gemm[0,0]);
    writeln(gemm[1,0]);
}


template isBlasType(T){
    enum isBlasType = is(T == float) || is(T == cfloat) || is(T == double) || is(T == cdouble);
}

template isVector(T){
    static if(__traits(compiles,{
                            T a;
                            int s = a.dim;
                            alias T.elementType U;
                            Vector!U b = a.evalExpr;
                        }))
        enum isVector = true;
    else
        enum isVector = false;
}
unittest{
    static assert(isVector!(Vector!double));
}

unittest{
    static assert(isVector!(Vector!double));
}

template isVectorExpr(T){
    enum isVectorExpr = isVector!T &&
                    __traits(compiles,{
                        T a;
                        alias T.argsType AT;
                        static assert(is(typeof(T.operator) : string));
                        
                        AT b = a.args;
                    });
}

unittest{
    auto a = Vector!double(3,3);
    static assert(isVectorExpr!(typeof(a * 3.0)));
}

template isMatrix(T){
    static if(__traits(compiles,{
                            T a;
                            int r = a.rows;
                            int c = a.cols;
                            alias T.elementType U;
                            Matrix!U b = a.evalExpr;
                        }))
        enum isMatrix = true;
    else
        enum isMatrix = false;
}

unittest{
    static assert(isMatrix!(Matrix!double));
}

template isMatrixExpr(T){
    enum isMatrixExpr = isMatrix!T && __traits(compiles,{
                                T a;
                                
                                alias T.argsType U;
                                auto b = a.args;
                                string op = T.operator;
                                static assert(is(typeof(T.operator) : string));
                            });
}

unittest{
    foreach(T; AllBlasType){
        auto a = Matrix!T(3,3), b = Matrix!T(3,3);
        static assert(!isMatrixExpr!(typeof(a)));
        static assert(isMatrixExpr!(typeof(a+b)));
        static assert(isMatrixExpr!(typeof(a-b)));
        static assert(isMatrixExpr!(typeof(a*b)));
    }
}

template isSpecialMatrix(T){
    enum isSpecialMatrix = isMatrix!T 
                            && !is(Matrix!(T.elementType) == T)
                            && __traits(compiles, {
                                auto b = cast(Matrix!(T.elementType))(T.init);
                            });
}

template isNormalMatrix(T){
    enum isNormalMatrix = isMatrix!T && !isSpecialMatrix!T;
}
unittest{
    static assert(isNormalMatrix!(Matrix!double));
    static assert(isNormalMatrix!(MatrixExpr!("scal" , cdouble, cdouble, Matrix!cdouble)));
    
    
    
    auto m = Matrix!double(2,2);
    m[0,0] = 2; m[0,1] = 1;
    m[1,0] = 5; m[1,1] = 3;
    
    //auto inv = m.getrf()[0..2].getri()[0];
    //auto inv = getri(getrf(m)[0..2]);
    auto inv = (m.getrf()[0..2]).getri;
    writeln(inv);
    //writeln(m * inv);
    writeln((m * inv[0]).evalExpr);
}

///その行列が転置な行列(isMatrixExprかつoperator == "transpose")ならtrue
template isTransposedMatrix(T){
    static if(isMatrixExpr!T && T.operator == "transposed")
        enum isTransposedMatrix = true;
    else
        enum isTransposedMatrix = false;
    //enum isTransposedMatrix = isMatrixExpr!T && T.operator == "transpose";
}

///その行列が共役行列ならtrue
template isConjugatedMatrix(T){
    static if(isMatrixExpr!T && T.operator == "conjugate")
        enum isConjugatedMatrix = true;
    else
        enum isConjugatedMatrix = false;
}

///その行列が共役転置行列ならtrue
template isConjTransMatrix(T){
    static if((isTransposedMatrix!T && isConjugatedMatrix!(T.argsType[0]))
            || (isConjugatedMatrix!T && isTransposedMatrix!(T.argsType[0])))
        enum isConjTransMatrix = true;
    else
        enum isConjTransMatrix = false;
}


