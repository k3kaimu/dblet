/** BLAS Level 1 についてExpression Templateを作るモジュールです
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.evalexpr.blas.lv1;

import ablaslv1 = dblet.blas.lv1;
import ablaslv2 = dblet.blas.lv2;
import ablaslv3 = dblet.blas.lv3;
import dblet.vector;
import dblet.traits;

mixin template VectorOpExpr_axpy(){
    static assert(argsType.length == 3);
    static assert(isVector!(argsType[1]));
    static assert(isVector!(argsType[2]));
    //static assert(is(argsType[1].elementType == argsType[2].elementType));
    //static assert(is(argsType[0] : argsType[1].elementType));
    
    
    @property
    Vector!E evalExpr(){
        return ablaslv1.axpy(args);
    }
    
    E opIndex(int i)
    in{
        assert(i < this.dim);
    }
    body{
        return args[0] * args[1][i] + args[2][i];
    }
    
    mixin(opBinAddVV);
    mixin(opBinSubVV);
    mixin(opBinMulVS);
}

mixin template MatrixOpExpr_axpy(){
    static assert(argsType.length == 3);
    static assert(isMatrix!(argsType[1]));
    static assert(isMatrix!(argsType[2]));
    
    @property
    Matrix!E evalExpr(){
        return ablaslv1.axpy(args);
    }
    
    E opIndex(int r, int c)
    in{
        assert(r < this.rows);
        assert(c < this.cols);
    }
    body{
        return args[0] * args[1][r,c] + args[2][r,c];
    }
    
    mixin(opBinAddMM);
    mixin(opBinSubMM);
    mixin(opBinMulMS);
    mixin(opBinMulMV);
    mixin(opBinMulMM);
}

mixin template VectorOpExpr_scal(){
    static assert(argsType.length == 2);
    static assert(isBlasType!(argsType[0]));
    static assert(isVector!(argsType[1]));
    //static assert(is(argsType[0] : argsType[1].elementType));
    
    @property
    Vector!E evalExpr(){
        return ablaslv1.scal(args);
    }
    
    E opIndex(int i)
    in{
        assert(i < this.dim);
    }
    body{
        return args[0] * args[1][i];
    }
    
    typeof(this) opUnary(string s : "-")(){
        return typeof(this)(this.dim, -args[0], args[1]);
    }
    
    typeof(this) opOpAssign(string s : "*")(E src)
    {
        this.args[0] *= src;
        return this;
    }
    
    auto opBinary(string s : "+", V)(V src)
    if(isVector!V && is(V.elementType == E))
    in{
        assert(src.dim == this.dim);
    }
    body{
        static if(__traits(compiles, V.init.opBinaryRight!"-"(VectorExpr!(Op, E, argsType).init)))
            return src.opBinaryRight!"-"(src);
        else
            return vectorExpr!("axpy", E)(this.dim, this.args, src);
    }
    
    
    mixin(opBinSubVV);
    
    auto opBinary(string s : "*")(E src)
    {
        return vectorExpr!("scal", E)(this.dim, this.args[0] * src, this.args[1]);
    }
    
}

mixin template MatrixOpExpr_scal(){
    static assert(argsType.length == 2);
    static assert(isMatrix!(argsType[1]));
    //static assert(is(argsType[0] == ArgT[1].elementType));
    
    @property
    Matrix!E evalExpr(){
        return ablaslv1.scal(args);
    }
    
    E opIndex(int r, int c)
    in{
        assert(r < this.rows);
        assert(c < this.cols);
    }
    body{
        return args[0] * args[1][r,c];
    }
    
    typeof(this) opUnary(string s : "-")(){
        return typeof(this)(this.rows, this.cols, -this.args[0], this.args[1]);
    }
    
    typeof(this) opOpAssign(string s : "*")(E src)
    {
        this.args[0] *= src;
        return this;
    }
    
    auto opBinary(string s : "+", M)(M src)
    if(isNormalMatrix!M && is(M.elementType == E))
    in{
        assert(src.rows == this.rows);
        assert(src.cols == this.cols);
    }
    body{
        static if(__traits(compiles, M.init.opBinaryRight!"+"(MatrixExpr!(Op, E, argsType).init)))
            return src.opBinaryRight!"+"(this);
        else
            return matrixExpr!("axpy", E)(this.rows, this.cols, this.args, src);
    }
    
    
    auto opBinary(string s : "-", M)(M src)
    if(isNormalMatrix!M && is(M.elementType == E))
    in{
        assert(src.rows == this.rows);
        assert(src.cols == this.cols);
    }
    body{
        static if(__traits(compiles, M.init.opBinaryRight!"-"(MatrixExpr!(Op, E, argsType).init)))
            return src.opBinaryRight!"-"(this);
        else static if(__traits(compiles, {auto a = M.init.opUnary!"-"();}))
            return matrixExpr!("axpy", E)(this.rows, this.cols, this, -src);
        else
            return matrixExpr!("axpy", E)(this.rows, this.cols, this, src * cast(E)-1);
    }
    
    
    auto opBinary(string s : "*", M)(M src)
    if(isNormalMatrix!M && is(M.elementType == E))
    in{
        assert(this.cols == src.rows);
    }
    body{
        static if(isMatrixExpr!M && M.operator == "scal")
            return matrixExpr!("gemm", E)(this.rows, src.cols, src.args[0] * this.args[0], this.args[1], src.args[1]);
        else
            return matrixExpr!("gemm", E)(this.rows, src.cols, this.args[0], this.args[1], src);
    }
    
    
    auto opBinary(string s : "*", V)(V src)
    if(isVector!V && is(V.elementType == E))
    in{
        assert(this.cols == src.dim);
    }
    body{
        static if(isMatrixExpr!V && V.operator == "scal")
            return vectorExpr!("gemv", E)(this.rows, src.args[0] * this.args[0], this.args[1], src.args[1]);
        else
            return vectorExpr!("gemv", E)(this.rows, this.args[0], this.args[1], src);
    }
    
    
    auto opBinary(string s : "*")(E src)
    {
        return matrixExpr!("scal", E)(this.rows, this.cols, this.args[0] * src, this.args[1]);
    }
    
}
