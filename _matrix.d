module dblet.matrix;

import dblet.evalexpr.blas.lv1;
import dblet.evalexpr.blas.lv2;
import dblet.evalexpr.blas.lv3;
//import dblet.evalexpr.lapack;
import dblet.vector;
import dblet.traits;

version(unittest){
    import std.algorithm;
    import std.stdio;
}

struct Matrix(T){
public:
    int _rows;
    int _cols;
    
    T[] array;
    
    alias T elementType;
    
    ///コンストラクタ
    this(U)(U r, U c, T[] arr){
        array = arr;
        _rows = r;
        _cols = c;
    }
    
    ///ditto
    this(U)(U r, U c){
        _rows = r;
        _cols = c;
        array.length = r*c;
    }
    
    @property
    typeof(this) evalExpr(){return this.dup;}
    
    @property
    typeof(this) dup(){return typeof(this)(_rows, _cols, array.dup);}
    
    @property
    int cols(){return _cols;}
    
    @property
    int rows(){return _rows;}
    
    ref T opIndex(int i, int j)
    in{
        assert(_rows > i);
        assert(_cols > j);
    }
    body{
        return array[j*_rows + i];
    }
    
    
    T opIndexAssign(T src, int i, int j)
    in{
        assert(_rows > i);
        assert(_cols > j);
    }
    body{
        return array[j*_rows + i] = src;
    }
    
    unittest{
        auto a = Matrix!double(2,2);
        a.array[] = 0;
        a[0,0] = 1;a[0,1] = 2;
        a[1,0] = 3;a[1,1] = 4;
        assert(equal(a.array, [1,3,2,4]));
    }
    
    /*///コピーする
    typeof(this) opAssign(MType)(MType src)if(isMatrix!MType && is(MType.elementType == T)){
        static if(isMatrixExpr!MType)
            array = src.evalExpr.array;
        else
            array = src.array.dup;
            
        return this;
    }*/
    
    auto opBinary(string s : "+", MType)(MType src)if(isMatrix!MType && is(MType.elementType == T))
    in{
        assert(src._rows == this._rows);
        assert(src._cols == this._cols);
    }
    body{
        static if(isMatrixExpr!MType && MType.operator == "scal")
            return matrixExpr!("axpy", T)(this._rows, this._cols, src.args, this.dup);
        else static if(isMatrixExpr!MType && MType.operator == "gemm" && MType.argsType.length == 3)
            return matrixExpr!("gemm", T)(this._rows, this._cols, src.args, cast(T)(1), this);
        else static if(isMatrixExpr!MType)
            return matrixExpr!("axpy", T)(this._rows, this._cols, cast(T)(1), this.dup, src);
        else
            return matrixExpr!("axpy", T)(this._rows, this._cols, cast(T)(1), this.dup, src.dup);
    }
    
    
    auto opBinary(string s : "-", MType)(MType src)if(isMatrix!MType && is(MType.elementType == T))
    in{
        assert(src._rows == this._rows);
        assert(src._cols == this._cols);
    }
    body{
        static if(isMatrixExpr!MType && MType.operator == "scal")
            return matrixExpr!("axpy", T)(this._rows, this._cols, src.args[0] * -1, src.args[1], this.dup);
        else static if(isMatrixExpr!MType && MType.operator == "gemm" && MType.argsType.length == 3)
            return matrixExpr!("gemv", T)(this._rows, this._cols, src.args[0] * -1, src.args[1..$], cast(T)(1), this.dup);
        else static if(isMatrixExpr!MType)
            return matrixExpr!("axpy", T)(this._rows, this._cols, cast(T)(-1), src, this.dup);
        else
            return matrixExpr!("axpy", T)(this._rows, this._cols, cast(T)(-1), src.dup, this.dup);
    }
    
    
    auto opBinary(string s : "*")(T src)
    {
        return matrixExpr!("scal", T)(this._rows, this._cols, src, this.dup);
    }
    
    auto opBinaryRight(string s : "*")(T src)
    {
        return matrixExpr!("scal", T)(this._rows, this._cols, src, this.dup);
    }
    
    
    auto opBinary(string s : "*", VType)(VType src)
    if(isVector!VType && is(VType.elementType == T))
    in{
        assert(this._cols == src._dim);
    }
    body{
        static if(isVectorExpr!VType && VType.operator == "scal")
            return vectorExpr!("gemv", T)(this.rows, src.args[0], this.dup, src.args[1]);
        else static if(isVectorExpr!VType)
            return vectorExpr!("gemv", T)(this.rows, cast(T)(1), this.dup, src);
        else
            return vectorExpr!("gemv", T)(this.rows, cast(T)(1), this.dup, src.dup);
    }
    
    
    auto opBinary(string s : "*", MType)(MType src)
    if(isNormalMatrix!MType && is(MType.elementType == T))
    in{
        assert(this._cols == src._rows);
    }
    body{
        static if(isMatrixExpr!MType && MType.operator == "scal")
            return matrixExpr!("gemm", T)(this._rows, src._cols, src.args[0], this.dup, src.args[1]);
        else static if(isMatrixExpr!MType)
            return matrixExpr!("gemm", T)(this._rows, src._cols, cast(T)(1), this.dup, src);
        else
            return matrixExpr!("gemm", T)(this._rows, src._cols, cast(T)(1), this.dup, src.dup);
    }
}

///線形代数 MatrixのExpression Template実装
struct MatrixExpr(alias Op, E, ArgT...){
package:
    int _rows;
    int _cols;
    argsType args;

public:
    static if(is(typeof(Op) : string))
        alias Op operator;
    alias E elementType;
    alias ArgT argsType;
    
    this(int r_size, int c_size, argsType src)
    {
        _rows = r_size;
        _cols = c_size;
        
        foreach(i, AType; ArgT)
            args[i] = src[i];
    }
    
    @property int rows(){return _rows;}
    
    @property int cols(){return _cols;}
    
    //mixin MatrixOpExpr!(Op);
    static if(is(typeof(Op) : string)){
        //pragma(msg, "MatrixOpExpr_" ~ Op);
        mixin("mixin MatrixOpExpr_" ~ Op ~ ";");
    }else{
        //pragma(msg, Op);
        mixin Op;
    }
}

///ditto
auto matrixExpr(string Op, E, ArgT...)(int r, int c, ArgT src)
{
    return MatrixExpr!(Op, E, ArgT)(r, c, src);
}

package enum opBinAddMM = q{
    auto opBinary(string s : "+", M)(M src)
    if(isNormalMatrix!M && is(M.elementType == E))
    in{
        assert(this.rows == src.rows);
        assert(this.cols == src.cols);
    }
    body{
        static if(__traits(compiles, {auto a = M.init.opBinaryRight!"+"(MatrixExpr!(Op, E, ArgT).init);}))
            return src.opBinaryRight!"+"(this);
        else static if(isMatrixExpr!M && M.operator == "scal")
            return matrixExpr!("axpy", E)(this.rows, this.cols, src.args, this);
        else
            return matrixExpr!("axpy", E)(this.rows, this.cols, cast(E)1, this, src);
    }
};

package enum opBinSubMM = q{
    auto opBinary(string s : "-", M)(M src)
    if(isNormalMatrix!M && is(M.elementType == E))
    in{
        assert(this.rows == src.rows);
        assert(this.cols == src.cols);
    }
    body{
        static if(__traits(compiles, {auto a = M.init.opBinaryRight!"-"(MatrixExpr!(Op, E, ArgT).init);}))
            return src.opBinaryRight!"-"(this);
        else
            return matrixExpr!("axpy", ET)(this.rows, this.cols, cast(E)-1, src, this);
    }
};

package enum opBinMulMS = q{
    auto opBinary(string s)(E src)if(s == "*" || s == "/")
    {
        return matrixExpr!("scal", E)(this.rows, this.cols, mixin(s == "*" ? "src" : "1/src"), this);
    }
    
    auto opBinaryRight(string s : "*")(E src)
    {
        return matrixExpr!("scal", E)(this.rows, this.cols, src, this);
    }
    
};

package enum opBinMulMV = q{
    auto opBinary(string s : "*", VType)(VType src)
    if(isVector!VType && is(VType.elementType == E))
    in{
        assert(this._cols == src._dim);
    }
    body{
        static if(isVectorExpr!VType && VType.operator == "scal")
            return vectorExpr!("gemv", E)(src._dim, src.args[0], this, src.args[1]);
        else static if(isVectorExpr!VType)
            return vectorExpr!("gemv", E)(src._dim, cast(E)1, this, src);
        else
            return vectorExpr!("gemv", E)(src._dim, cast(E)1, this, src.dup);
    }
};

package enum opBinMulMM = q{
    auto opBinary(string s : "*", MType)(MType src)if(isMatrix!MType && is(MType.elementType == E))
    in{
        assert(this._cols == src._rows);
    }
    body{
        static if(isMatrixExpr!MType && MType.operator == "scal")
            return matrixExpr!("gemm", E)(this._rows, src._cols, src.args[0], this, arc.args[1]);
        else static if(isMatrixExpr!MType)
            return matrixExpr!("gemm", E)(this._rows, src._cols, cast(E)1, this, src);
        else
            return matrixExpr!("gemm", E)(this._rows, src._cols, cast(E)1, this, src.dup);
    }
};

