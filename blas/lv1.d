/** BLAS Level 1 についてライブラリとBLASを結びつけるためのモジュールです
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.blas.lv1;

import std.typecons : tuple;

//import scid.bindings.lapack.lapack;
static import dblas = scid.bindings.blas.dblas;

import dblet.traits;

version(unittest){
    import std.math;
    
    import dblet.vector;
}

/++
ベクトルの各要素の和.
+/
auto asum(L)(L vec)if(isVector!L || isMatrix!L)
{
    return dblas.asum(vec.dim, vec.evalExpr.array.ptr, 1);
}
unittest{
    assert(55 == asum(Vector!double([1,2,3,4,5,6,7,8,9,10])));
    assert(55 == asum(Vector!cdouble([1+2i,3+4i,5+6i,7+8i,9+10i])));
}


///axpy(y := a*x +y)の実装
auto axpy(T, LX, LY)(T alpha, LX x, LY y)
if(((isVector!LX && isVector!LX) || (isMatrix!LX && isMatrix!LX))
        && is(LX.elementType == LY.elementType)
        && is(T == LX.elementType))
in{
    static if(isMatrix!LX){
        assert(x.rows == y.rows);
        assert(x.cols == y.cols);
    }else
        assert(x.dim == y.dim);
}
body{
    auto _y = y.evalExpr;
    
    static if(isNormalMatrix!LX)
        dblas.axpy(x.rows * x.cols, alpha, x.evalExpr.array.ptr, 1, _y.array.ptr, 1);
    else
        dblas.axpy(x.dim, alpha, x.evalExpr.array.ptr, 1, _y.array.ptr, 1);
    
    return _y;
}


///copy(y = x)の実装
void copy(LX, LY)(LX x, ref LY y)
if((isVector!LX && isVector!LY && !isVectorExpr!LY)
    || (isMatrix!LX && isMatrix!LY && !isMatrixExpr!LY))
in{
    static if(isMatrix!LX){
        assert(x.rows == y.rows);
        assert(x.cols == y.cols);
    }else
        assert(x.dim == y.dim);
}
body{
    static if(isNormalMatrix!LX)
        dblas.copy(x.rows * x.cols, x.evalExpr.array.ptr, 1, y.array.ptr, q);
    else
        dblas.copy(x.dim, x.evalExpr.array.ptr, 1, y.array.ptr, q);
}

/++
ベクトルとベクトルのドット積(内積,スカラー積とも)を計算します
+/
VX.elementType dot(VX, VY)(VX x, VY y)
if(is(VX.elementType == VY.elementType))
in{
    assert(x.dim == y.dim);
}
body{
    static if(is(VX.elementType : real))
        alias dblas.dot _dot;
    else
        alias dblas.dotu _dot;
    
    static if(isVectorExpr!VX && VX.operator == "conjugate"){
        static if(isVectorExpr!VY)
            return dblas.dotc(x.dim, x.args[0].evalExpr.array.ptr, 1, y.evalExpr.array.ptr, 1);
        else
            return dblas.dotc(x.dim, x.args[0].evalExpr.array.ptr, 1, y.array.ptr, 1);
    }else static if(isVectorExpr!VY && VX.operator == "conjugate"){
        static if(isVectorExpr!VX)
            return dblas.dotc(x.dim, y.args[0].evalExpr.array.ptr, 1, x.evalExpr.array.ptr, 1);
        else
            return dblas.dotc(x.dim, y.args[0].evalExpr.array.ptr, 1, x.array.ptr, 1);
    }else
        return _dot(x.dim, x.evalExpr.array.ptr, 1, y.evalExpr.array.ptr, 1);
}
unittest{
    auto a = Vector!double([1,2,3]),b = Vector!double([2,3,4]);
    assert(20 == dot(a, b));
}

/++
ベクトルのユークリッドノルムを計算します
+/
V.elementType nrm2(V)(V vec)
if(isVector!V)
{
    return dblas.nrm2(vec.dim, vec.evalExpr.array.ptr, 1);
}
unittest{
    import std.stdio;
    auto a = Vector!double([3,4]),b = Vector!double([5,12]);
    assert(approxEqual(nrm2(a), 5, 1e-2));
    assert(approxEqual(nrm2(b), 13, 1e-2));
}


///rotの実装
auto rot(VX, VY, T)(VX vecx, VY vecy, T c, T s)
if(is(VX.elementType == VY.elementType) && is(T == VX.elementType))
in{
    assert(vecx.dim == vecy.dim);
}
body{
    dblas.rot(x.dim, x.array.ptr, 1, y.array.ptr, 1, c, s);
    
    return tuple(x, y);
}

alias dblas.rotg rotg;

///rotmの実装
auto rotm(VX, VY, T, M)(VX vecx, VY vecy, T param1, M pmat)
if(isVector!VX && isVector!VY && isMatrix!M
    && is(VX.elementType == VY.elementType)
    && is(VX.elementType == M.elementType)
    && is(T == M.elementType))
in{
    assert(vecx.dim == vecy.dim);
    assert(pmat.rows == 2);
    assert(pmat.cols == 2);
    assert(param1 == -2 || param1 == -1 || param1 == 0 || param1 == 1);
}
body{
    auto dstx = x.evalExpr, dsty = y.evalExpr, param = param1 ~ pmat.evalExpr.array;
    dblas.rot(x.dim, dstx.array.ptr, 1, dsty.array.ptr, 1, param.ptr);
    
    return tuple(dstx, dsty, param[0], Matrix!M.elementType(2, 2, param[1..$]));
}

///rotmgの実装
auto rotmg(T)(T d1, T d2, T x1, T y1)if(isBlasType!T){
    T[] arr = new T[5];
    
    dblas.rotmg(d1, d2, x1, y1, arr.ptr);
    
    return tuple(arr[0], Matrix!T(2, 2, arr[1..$]));
}

///scal(x := a*x)のExpression Templateの実装
auto scal(T, L)(T alpha, L x)
if((isVector!L || isMatrix!L) && is(T == L.elementType))
{
    alias L.elementType E;
    
    auto _x = x.evalExpr;
    
    static if(isNormalMatrix!L)
        dblas.scal(x.rows * x.cols, alpha, _x.array.ptr, 1);
    else
        dblas.scal(x.dim, alpha, _x.array.ptr, 1);
    
    return _x;
}


///swap
void swap(L)(ref L x, ref L y)
if((isVector!L && !isVectorExpr!L) || (isMatrix!L && !isMatrixExpr!L))
in{
    assert(x.dim == y.dim);
}
body{
    dblas.swap(x.dim, x.array.ptr, 1, y.array.ptr, 1);
}


///iamax
int iamax(V)(V x)
if(isVector!V)
{
    static if(__traits(compiles, dblas.isamax(1, V.init.array.ptr, 1)))
        return isamax(x.dim, x.evalExpr.array.ptr, 1);
    else static if(__traits(compiles, dblas.idamax(1, V.init.array.ptr, 1)))
        return idamax(x.dim, x.evalExpr.array.ptr, 1);
    else static if(__traits(compiles, dblas.icamax(1, V.init.array.ptr, 1)))
        return icamax(x.dim, x.evalExpr.array.ptr, 1);
    else static if(__traits(compiles, dblas.izamax(1, V.init.array.ptr, 1)))
        return izamax(x.dim, x.evalExpr.array.ptr, 1);
    else
        static assert(0, "Matching Error");
}

///iamin
int iamin(V)(V x)
if(isVector!V)
{
    static if(__traits(compiles, dblas.isamin(1, V.init.array.ptr, 1)))
        return isamin(x.dim, x.evalExpr.array.ptr, 1);
    else static if(__traits(compiles, dblas.idamin(1, V.init.array.ptr, 1)))
        return idamin(x.dim, x.evalExpr.array.ptr, 1);
    else static if(__traits(compiles, dblas.icamin(1, V.init.array.ptr, 1)))
        return icamin(x.dim, x.evalExpr.array.ptr, 1);
    else static if(__traits(compiles, dblas.izamin(1, V.init.array.ptr, 1)))
        return izamin(x.dim, x.evalExpr.array.ptr, 1);
    else
        static assert(0, "Matching Error");
}

