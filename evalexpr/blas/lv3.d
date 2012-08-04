/** BLAS Level 3 についてExpression Templateを作るモジュールです
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.evalexpr.blas.lv3;

static import ablas = dblet.blas.lv3;
import dblet.matrix;
import dblet.traits;


mixin template MatrixOpExpr_gemm(){
    static assert(argsType.length == 3 || argsType.length == 5);
    static assert(isBlasType!(argsType[0]));
    static assert(isNormalMatrix!(argsType[1]));
    static assert(isNormalMatrix!(argsType[2]));
    
    static if(argsType.length == 5){
        static assert(isBlasType!(argsType[3]));
        static assert(isNormalMatrix!(argsType[4]));
    }
    
    @property
    Matrix!E evalExpr(){
        return ablaslv3.gemm(this.args);
    }
    
    
    E opIndex(int r, int c)
    in{
        assert(r < this.rows);
        assert(c < this.cols);
    }
    body{
        E sum = cast(E)0;
        
        for(int i = 0; i < args[1].cols; ++i)
            sum += args[1][r, i] * args[2][i, c];
        
        static if(argsType.length == 3)
            return sum * args[0];
        else
            return sum * args[0] + args[3] * args[4][r, c];
    }
    
    
    typeof(this) opUnary(string s : "-")(){
        static if(argsType.length == 3)
            return matrixExpr!("gemm", E)(this.rows, this.cols, -this.args[0], this.args[1..3]);
        else
            return matrixExpr!("gemm", E)(this.rows, this.cols, -this.args[0], this.args[1..3], -this.args[3], this.args[4]);
    }
    
    
    typeof(this) opOpAssign(string s : "*")(E src)
    {
        this.args[0] *= src;
        static if(argsType.length == 5)
            this.args[3] *= src;
        return this;
    }
    
    
    static if(argsType.length == 3){
        auto opBinaryRight(string s : "+", M)(M src)
        if(isNormalMatrix!M && is(M.elementType == E))
        in{
            assert(this.rows == src.rows);
            assert(this.cols == src.cols);
        }
        body{
            static if(isMatrixExpr!M && M.operator == "scal")
                return matrixExpr!("gemm", E)(this.rows, this.cols, this.args, src.args);
            else
                return matrixExpr!("gemm", E)(this.rows, this.cols, this.args, cast(E)1, src);
        }
        
        auto opBinaryRight(string s : "-", M)(M src)
        if(isNormalMatrix!M && is(M.elementType == E))
        in{
            assert(this.rows == src.rows);
            assert(this.cols == src.cols);
        }
        body{
            static if(isMatrixExpr!M && M.operator == "scal")
                return matrixExpr!("gemm", E)(this.rows, this.cols, (-this).args, src.args);
            else
                return matrixExpr!("gemm", E)(this.rows, this.cols, (-this).args, cast(E)1, src.args);
        }
    }
    
    auto opBinary(string s : "+", M)(M src)
    if(isNormalMatrix!M && is(M.elementType == E))
    in{
        assert(this.rows == src.rows);
        assert(this.cols == src.cols);
    }
    body{
        static if(argsType.length == 3){
            static if(isMatrixExpr!M && M.operator == "scal")
                return matrixExpr!("gemm", E)(this.rows, this.cols, this.args, src.args);
            else
                return matrixExpr!("gemm", E)(this.rows, this.cols, this.args, cast(E)1, src);
        }else static if(__traits(compiles, M.init.opBinaryRight!"+"(MatrixExpr!(Op, E, argsType).init)))
            return src.opBinaryRight!"+"(this);
        else static if(isMatrixExpr!M && M.operator == "scal")
            return matrixExpr!("axpy", E)(this.rows, this.cols, src.args, this);
        else
            return matrixExpr!("axpy", E)(this.rows, this.cols, cast(E)1, this, src);  
    }
    
    auto opBinary(string s : "-", M)(M src)
    if(isNormalMatrix!M && is(M.elementType == E))
    in{
        assert(this.rows == src.rows);
        assert(this.cols == src.cols);
    }
    body{
        static if(argsType.length == 3){
            static if(isMatrixExpr!M && M.operator == "scal")
                return matrixExpr!("gemm", E)(this.rows, this.cols, this.args, (-src).args);
            else
                return matrixExpr!("gemm", E)(this.rows, this.cols, this.args, cast(E)-1, src);
        }else static if(__traits(compiles, M.init.opBinaryRight!"-"(MatrixExpr!(Op, E, argsType).init)))
            return src.opBinaryRight!"-"(this);
        else static if(isMatrixExpr!M && M.operator == "scal")
            return matrixExpr!("axpy", E)(this.rows, this.cols, (-src).args, this);
        else
            return matrixExpr!("axpy", E)(this.rows, this.cols, cast(E)-1, src, this);  
    }
    
    auto opBinary(string s : "*")(E src)
    {
        static if(argsType.length == 3)
            return matrixExpr!("gemm", E)(this.rows, this.cols, this.args[0] * src, this.args[1..3]);
        else
            return matrixExpr!("gemm", E)(this.rows, this.cols, this.args[0] * src, this.args[1..3], this.args[3] * src, this.args[4]);
    }
    
    mixin(opBinMulMV);
    mixin(opBinMulMM);
    
}
