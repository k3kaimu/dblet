/** BLAS Level 2 について本ライブラリとBLASを結びつけるためのモジュールです
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.blas.lv2;

static import dblas = scid.bindings.blas.dblas;
import dblet.traits;
import dblet.vector;

/**?gbmv:一般バンド行列とベクトルの積(に他のベクトルを加える)。$(BR)
式. y := αAx + βy $(BR)
BLASではtransで共役転置,転置,通常のなかから選べるが、この実装では、たとえば$(BR)
y = gbmv(alpha, a.transposed(), x, beta, y);$(BR)
により、内部で転置'T'を呼び出したり、$(BR)
y = gbmv(alpha, a.conjugated().transposed(), x, beta, y);$(BR)
により、内部で共役転置'C'を呼び出したりする。$(BR)
*/
Vector!T gbmv(T, Mb, VX)(T alpha, Mb a, VX x)
if(isBlasType!T && isBandMatrix!Mb && isVector!VX
    && is(T == Mb.elementType) && is(T == VX.elementType))
in{
    assert(a.cols == x.dim);
}
body{
    auto y = Vector!T(a.rows);
    
    static if(isConjTransMatrix!Mb)
        dblas.gbmv('C', a.cols, a.rows, a.ku, a.kl, alpha, a.args[0].args[0].evalExpr, a.kl + a.ku + 1, x.evalExpr.array.ptr, 1, 0, y.array.ptr, 1);
    else static if(isTransposedMatrix!Mb)
        dblas.gbmv('T', a.cols, a.rows, a.ku, a.kl, alpha, a.args[0].evalExpr, a.kl + a.ku + 1, x.evalExpr.array.ptr, 1, 0, y.array.ptr, 1);
    else
        dblas.gbmv('N', a.rows, a.cols, a.kl, a.ku, alpha, a.args[0].evalExpr, a.kl + a.ku + 1, x.evalExpr.array.ptr, 1, 0, y.array.ptr, 1); 
    
    return y;
}

///ditto
Vector!T gbmv(T, Mb, VX)(T alpha, Mb a, VX x, T beta, VY y)
if(isBlasType!T && isBandMatrix!Mb && isVector!VX && isVector!VY
    && is(T == Mb.elementType) && is(T == VX.elementType) && is(T == VY.elementType))
in{
    assert(a.cols == x.dim);
}
body{
    auto _y = y.evalExpr;
    
    static if(isConjTransMatrix!Mb)
        dblas.gbmv('C', a.cols, a.rows, a.ku, a.kl, alpha, a.args[0].args[0].evalExpr, a.kl + a.ku + 1, x.evalExpr.array.ptr, 1, beta, _y.array.ptr, 1);
    else static if(isTransposedMatrix!Mb)
        dblas.gbmv('T', a.cols, a.rows, a.ku, a.kl, alpha, a.args[0].evalExpr, a.kl + a.ku + 1, x.evalExpr.array.ptr, 1, beta, _y.array.ptr, 1);
    else
        dblas.gbmv('N', a.rows, a.cols, a.kl, a.ku, alpha, a.args[0].evalExpr, a.kl + a.ku + 1, x.evalExpr.array.ptr, 1, beta, _y.array.ptr, 1); 
    
    return _y;
}


/**?gemv:一般行列とベクトルの積(に他のベクトルを加える)。$(BR)
式. y := αAx + βy $(BR)
BLASではtransで共役転置,転置,通常のなかから選べるが、この実装では、たとえば$(BR)
y = gemv(alpha, a.transposed(), x, beta, y);$(BR)
により、内部で転置'T'を呼び出したり、$(BR)
y = gemv(alpha, a.conjugated().transposed(), x, beta, y);$(BR)
により、内部で共役転置'C'を呼び出したりする。$(BR)
*/
Vector!T gemv(T, M, VX)(T alpha, M a, VX x)
if(isBlasType!T && isNormalMatrix!M && isVector!VX
    && is(M.elementType == T) && is(VX.elementType == T))
in{
    assert(a.cols == x.dim);
}
body{
    auto y = Vector!T(a.rows);
    
    static if(isConjTransMatrix!M)
        dblas.gemv('C', a.cols, a.rows, alpha, a.args[0].args[0].evalExpr.array.ptr, a.cols, x.evalExpr.array.ptr, 1, 0, y.array.ptr, 1);
    else static if(isTransposedMatrix!M)
        dblas.gemv('T', a.cols, a.rows, alpha, a.args[0].evalExpr.array.ptr, a.cols, x.evalExpr.array.ptr, 1, 0, y.array.ptr, 1);
    else
        dblas.gemv('N', a.rows, a.cols, alpha, a.evalExpr.array.ptr, a.rows, x.evalExpr.array.ptr, 1, 0, y.array.ptr, 1);
    
    return y;
}

///ditto
Vector!T gemv(T, M, VX, VY)(T alpha, M a, VX x, T beta, VY y)
if(isBlasType!T && isNormalMatrix!M && isVector!VX && isVector!VY
    && is(M.elementType == T) && is(VX.elementType == T) && is(VY.elementType == T))
in{
    assert(a.cols == x.dim);
    assert(a.rows == y.dim);
}
body{
    auto _y = y.evalExpr;
    
    static if(isConjTransMatrix!M)
        dblas.gemv('C', a.cols, a.rows, alpha, a.args[0].args[0].evalExpr.array.ptr, a.cols, x.evalExpr.array.ptr, 1, beta, _y.array.ptr, 1);
    else static if(isTransposedMatrix!M)
        dblas.gemv('T', a.cols, a.rows, alpha, a.args[0].evalExpr.array.ptr, a.cols, x.evalExpr.array.ptr, 1, beta, _y.array.ptr, 1);
    else
        dblas.gemv('N', a.rows, a.cols, alpha, a.evalExpr.array.ptr, a.rows, x.evalExpr.array.ptr, 1, beta, _y.array.ptr, 1);
    
    return _y;
}


/**?ger or ?gerc or ?geru:ベクトルとベクトルの直積(に他の行列を加える)。$(BR)
式. a := αx * y_T + a
ただし、yがconjugated(z)な場合には
式. a := αx * conj(z_T) + a
となり、gercが呼び出される。
*/
Matrix!T ger(T, VX, VY)(T alpha, VX x, VY y)
if(isBlasType!T && isVector!VX && isVector!VY
    && is(T == VX.elementType) && is(T == VX.elementType))
{
    auto a = Matrix!T(x.dim, y.dim);
    
    dblas.ger(x.dim, y.dim, alpha, x.evalExpr.array.ptr, 1, y.evalExpr.array.ptr, 1, a.array.ptr, x.dim);
    
    return a;
}

///ditto
Matrix!T ger(T, VX, VY, M)(T alpha, VX x, VY y, M a)
if(isBlasType!T && isVector!VX && isVector!VY && isNormalMatrix!M
    && is(T == VX.elementType) && is(T == VX.elementType) && is(T == M.elementType))
in{
    assert(x.dim == a.rows);
    assert(y.dim == a.cols);
}
body{
    auto _a = a.evalExpr;
    
    static if(is(T : real))
        dblas.ger(a.rows, a.cols, alpha, x.evalExpr.array.ptr, 1, y.evalExpr.array.ptr, 1, _a.array.ptr, a.rows);
    else{
        static if(isVectorExpr!VY && VY.operator == "conjugated")
            dblas.gerc(a.rows, a.cols, alpha, x.evalExpr.array.ptr, 1, y.evalExpr.array.ptr, 1, _a.array.ptr, a.rows);
        else
            dblas.geru(a.rows, a.cols, alpha, x.evalExpr.array.ptr, 1, y.evalExpr.array.ptr, 1, _a.array.ptr, a.rows);
    }

    return a;
}




