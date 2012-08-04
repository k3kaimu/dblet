/** ベクトルを扱います
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.vector;

import scid.bindings.lapack.dlapack;
import scid.bindings.blas.dblas;

import dblet.matrix;
import dblet.traits;
import dblet.evalexpr.blas.lv1;
import dblet.evalexpr.blas.lv2;

import dblet.tuple;

version(unittest){
    import std.range;
    import std.algorithm;
    import std.stdio;
}

/**ベクトル型
* Example:
* ---------------------
void main(){
    Vector!double a;		//動的なベクトル
    
    assert(a.length == 0);
    
    a.length = 3;
    assert(a.length == 0);
}
* ---------------------
*/
struct Vector(T){
public:
    T[] array;
    alias T elementType;
    
    /** コンストラクタ
     * Example:
     * ----------
     * auto ary = [0.1,0.2,0.3];
     * 
     * auto va = Vector!double(ary);
     * auto vb = Vector!double(ary.ptr, ary.length);
     * ----------
     */
    this(T[] arr)
    {
        array = arr;
    }
    
    ///ditto
    this(T* p, int s)
    {
        array.length = s;
        for(int i = 0; i < s; ++i)
            array[i] = p[i];
    }
    
    ///ditto
    this(int s){
        array.length = s;
    }
    
    ///ditto
    this(int s, T init){
        array.length = s;
        array[] = init;
    }
    
    alias dim _dim;
    @property int dim(){return array.length;}
    
    @property typeof(this) evalExpr(){return this.dup;}
    
    @property typeof(this) dup(){return typeof(this)(array.dup);}
    
    ref T opIndex(int idx)
    in{
        assert(array.length > idx);
    }
    body{
        return array[idx];
    }
    
    
    T opIndexAssign(T src, int idx)
    in{
        assert(array.length > idx);
    }
    body{
        return (array[idx] = src);
    }
    
    /*
    ///コピーする
    typeof(this) opAssign(VType : typeof(this))(VType src){
        array = src.array.dup;
        return this;
    }
    
    typeof(this) opAssign(VType)(VType src){
        static if(isVectorExpr!VType)
            array = src.evalExpr.array;
        else
            array = src.array.dup;
            
        return this;
    }*/
    
//演算子オーバーロード
    auto opBinary(string s : "+", VType)(VType src)if(isVector!VType && is(T == VType.elementType))
    in{
        assert(this._dim == src._dim);
    }
    body{
        static if(isVectorExpr!VType && VType.operator == "scal")
            return vectorExpr!("axpy", T)(this._dim, src.args, this.dup);
        else static if(isVectorExpr!VType && VType.operator == "gemv" && VType.argsType.length == 3)
            return vectorExpr!("gemv", T)(this._dim, src.args, cast(T)1, this.dup);
        else static if(isVectorExpr!VType)
            return vectorExpr!("axpy", T)(this._dim, cast(T)1, this.dup, src);
        else
            return vectorExpr!("axpy", T)(this._dim, cast(T)1, this.dup, src.dup);
    }
    
    
    auto opBinary(string s : "-", VType)(VType src)if(isVector!VType && is(T == VType.elementType))
    in{
        assert(this._dim == src._dim);
    }
    body{
        static if(isVectorExpr!VType && VType.operator == "scal")
            return vectorExpr!("axpy", T)(this._dim, -src.args[0], src.args[1], this.dup);
        else static if(isVectorExpr!VType && VType.operator == "gemv" && VType.argsType.length == 3)
            return vectorExpr!("gemv", T)(this._dim, -src.args[0], src.args[1..$], cast(T)1, this.dup);
        else static if(isVectorExpr!VType)
            return vectorExpr!("axpy", T)(this._dim , cast(T)(-1), src, this.dup);
        else
            return vectorExpr!("axpy", T)(this._dim , cast(T)(-1), src.dup, this);
    }
    
    
    auto opBinary(string s : "*")(T src)
    {
        return vectorExpr!("scal", T)(this._dim, cast(T)(src), this.dup);
    }
    
    unittest{
        auto a = Vector!double(3), b = Vector!double(3);
        a.array[] = 10;
        b.array[] = 5;
        
        auto c = a * 3.0;
        static assert(typeof(c).operator == "scal");
        assert(equal(c.evalExpr.array, [30.0, 30.0, 30.0]));
        
        auto d = a + b;
        static assert(typeof(d).operator == "axpy");
        assert(equal(d.evalExpr.array, [15.0, 15.0, 15.0]));
        
        auto e = a - b;
        static assert(typeof(e).operator == "axpy");
        assert(equal(e.evalExpr.array, [5.0, 5.0, 5.0]));
    }
    
    auto opBinaryRight(string s : "*")(T src)
    {
        return vectorExpr!("scal", T)(this._dim, cast(T)(src), this.dup);
    }
}

///線形代数 VectorのExpression Template実装
struct VectorExpr(alias Op, E, ArgT...){
package:
    int _dim;
    argsType args;

public:
    static if(is(typeof(Op) : string))
        alias Op operator;
    alias E elementType;
    alias ArgT argsType;

    this(int size, ArgT src){
        _dim = size;
        foreach(i, AType; ArgT)
                args[i] = src[i];
    }
    
    @property int dim(){
        return _dim;
    }
    
    static if(is(typeof(Op) : string))
        mixin("mixin VectorOpExpr_" ~ Op ~ ";");
    else
        mixin Op;
}

///ditto
auto vectorExpr(string Op, E, ArgT...)(int s, ArgT src){
    return VectorExpr!(Op, E, ArgT)(s, src);
}

package enum opBinAddVV = q{
    auto opBinary(string s : "+", V)(V src)
    if(isVector!V && is(V.elementType == E))
    in{
        assert(this.dim == src.dim);
    }
    body{
        static if(__traits(compiles, {auto a = V.init.opBinaryRight!"+"(VectorExpr!(Op, E, ArgT).init);}))
            return src.opBinaryRight!"+"(this);
        else static if(isVectorExpr!V && V.operator == "scal")
            return vectorExpr!("axpy", E)(this.dim, src.args, this);
        else
            return vectorExpr!("axpy", E)(this.dim, cast(E)1, this, src);
    }
};

package enum opBinSubVV = q{
    auto opBinary(string s : "-", V)(V src)
    if(isVector!V && is(V.elementType == E))
    in{
        assert(this.dim == src.dim);
    }
    body{
        static if(__traits(compiles, V.init.opBinaryRight!"-"(VectorExpr!(Op, E, ArgT).init)))
            return src.opBinaryRight!"-"(this);
        else static if(isVectorExpr!V && V.operator == "scal")
            return vectorExpr!("axpy", E)(this.dim, -src.args[0], src.args[1], this);
        else
            return vectorExpr!("axpy", E)(this.dim, cast(E)-1, src, this);
    }
};

package enum opBinMulVS = q{
    auto opBinary(string s)(E src)if(s == "*" || s == "/")
    {
        return vectorExpr!("scal", E)(this.dim, mixin(s == "*" ? "src" : "1/src"), this);
    }
    
    auto opBinaryRight(string s : "*")(E src)
    {
        return vectorExpr!("scal", E)(this.dim, src, this);
    }
};

