/** BLAS Level 3 について本ライブラリとBLASを結びつけるためのモジュールです
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.blas.lv3;

import dblet.traits;
import dblet.matrix;
import dblas = scid.bindings.blas.dblas;

/**行列同士の積(に他の行列を加える)
式. C := αAB + C
*/
Matrix!T gemm(T, MA, MB)(T alpha, MA a, MB b)
if(isBlasType!T && isNormalMatrix!MA && isNormalMatrix!MB
    && is(MA.elementType == T) && is(MB.elementType == T) )
in{
    assert(a.cols == b.rows);
}
body{
    auto _c = Matrix!T(a.rows, b.cols);
    
    static if(isConjTransMatrix!MA){
        enum transa = 'C';
        auto _a = a.args[0].args[0];
    }else static if(isTransposedMatrix!MA){
        enum transa = 'T';
        auto _a = a.args[0];
    }else{
        enum transa = 'N';
        alias a _a;
    }
    
    static if(isConjTransMatrix!MB){
        enum transb = 'C';
        auto _b = b.args[0].args[0];
    }else static if(isTransposedMatrix!MB){
        enum transb = 'T';
        auto _b = b.args[0];
    }else{
        enum transb = 'N';
        alias b _b;
    }
    
    dblas.gemm(transa, transb, a.rows, b.cols, b.rows, alpha, _a.evalExpr.array.ptr, _a.rows, b.evalExpr.array.ptr, _b.rows, cast(T)0, _c.array.ptr, a.rows);
    return _c;
}


///ditto
Matrix!T gemm(T, MA, MB, MC)(T alpha, MA a, MB b, T beta, MC c)
if(isBlasType!T && isNormalMatrix!MA && isNormalMatrix!MB && isNormalMatrix!MC
    && is(MA.elementType == T) && is(MB.elementType == T) && is(MC.elementType == T))
in{
    assert(a.cols == b.rows);
    assert(a.rows == c.rows);
    assert(b.cols == b.cols);
}
body{
    auto _c = c.evalExpr;
    
    static if(isConjTransMatrix!MA){
        enum transa = 'C';
        auto _a = a.args[0].args[0];
    }else static if(isTransposedMatrix!MA){
        enum transa = 'T';
        auto _a = a.args[0];
    }else{
        enum transa = 'N';
        alias a _a;
    }
    
    static if(isConjTransMatrix!MB){
        enum transb = 'C';
        auto _b = b.args[0].args[0];
    }else static if(isTransposedMatrix!MB){
        enum transb = 'T';
        auto _b = b.args[0];
    }else{
        enum transb = 'N';
        alias b _b;
    }
    
    dblas.gemm(transa, transb, a.rows, b.cols, b.rows, alpha, _a.evalExpr.array.ptr, _a.rows, b.evalExpr.array.ptr, _b.rows, beta, _c.array.ptr, _c.rows);
    return _c;
}

