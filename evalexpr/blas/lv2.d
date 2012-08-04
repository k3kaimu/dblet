/** BLAS Level 2 についてExpression Templateを作るモジュールです
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.evalexpr.blas.lv2;

static import ablas = dblet.blas.lv2;
import dblet.vector;
import dblet.traits;


mixin template VectorOpExpr_gemv(){
    static assert(argsType.length == 3 || argsType.length == 5);
    static assert(isBlasType!(argsType[0]));
    static assert(isMatrix!(argsType[1]));
    static assert(isVector!(argsType[2]));
    
    static if(argsType.length == 5){
        static assert(isBlasType!(argsType[3]));
        static assert(isVector!(argsType[4]));
    }
    
    @property
    Vector!E evalExpr(){
        return ablaslv2.gemv(args);
    }
    
    E opIndex(int i)
    in{
        assert(i < this.dim);
    }
    body{
        E sum = cast(E)0;
        
        for(int j = 0; j < args[1].cols; ++j)
            sum += args[1][i, j] * args[2][j];
        
        static if(argsType.length == 3)
            return sum * args[0];
        else
            return sum * args[0] + args[3] * args[4][i];
    }
    
    typeof(this) opUnary(string s : "-")(){
        static if(argsType.length == 3)
            return typeof(this)(this.dim, -args[0], args[1..$]);
        else
            return typeof(this)(this.dim, -args[0], args[1..3], -args[3], args[4]);
    }
    
    typeof(this) opOpAssign(string s : "*")(E src)
    {
        this.args[0] *= src;
        static if(argsType.length == 5)
            this.args[3] *= src;
        return this;
    }
    
    static if(argsType.length == 3){
        auto opBinaryRight(string s : "+", V)(V src)
        if(isVector!V && is(V.elementType == E))
        in{
            assert(src.dim == this.dim);
        }
        body{
            static if(isVectorExpr!V && V.operator == "scal")
                return vectorExpr!("gemv", E)(this.dim, this.args, src.args);
            else
                return vectorExpr!("gemv", E)(this.dim, this.args, cast(E)1, src);
        }
        
        auto opBinaryRight(string s : "-", V)(V src)
        if(isVector!V && is(V.elementType == E))
        in{
            assert(src.dim == this.dim);
        }
        body{
            static if(isVectorExpr!V && V.operator == "scal")
                return vectorExpr!("gemv", E)(this.dim, this.args, -src.args);
            else
                return vectorExpr!("gemv", E)(this.dim, -this.args[0], this.args[1..3], cast(E)1, src);
        }
    }
    
    auto opBinary(string s : "+", V)(V src)
    if(isVector!V && is(V.elementType == E))
    in{
        assert(src.dim == this.dim);
    }
    body{
        static if(argsType.length == 3){
            static if(isVectorExpr!V && V.operator == "scal")
                return vectorExpr!("gemv", E)(this.dim, this.args, src.args);
            else
                return vectorExpr!("gemv", E)(this.dim, this.args, cast(E)1, src);
        }else static if(__traits(compiles, V.init.opBinaryRight!"+"(VectorExpr!(Op, E, argsType).init)))
            return src.opBinaryRight!"+"(this);
        else static if(isVectorExpr!V && V.operator == "scal")
            return vectorExpr!("axpy", E)(this.dim, src.args, this);
        else
            return vectorExpr!("axpy", E)(this.dim, cast(E)1, this, src);
    }
    
    
    auto opBinary(string s : "-", V)(V src)
    if(isVector!V && is(V.elementType == E))
    in{
        assert(src.dim == this.dim);
    }
    body{
        static if(argsType.length == 3){
            static if(isVectorExpr!V && V.operator == "scal")
                return vectorExpr!("gemv", E)(this.dim, this.args, (-src).args);
            else
                return vectorExpr!("gemv", E)(this.dim, this.args, cast(E)-1, src);
        }else static if(__traits(compiles, V.init.opBinaryRight!"-"(VectorExpr!(Op, E, argsType).init)))
            return src.opBinaryRight!"-"(this);
        else static if(isVectorExpr!V && V.operator == "scal")
            return vectorExpr!("axpy", E)(this.dim, (-src).args, this);
        else
            return vectorExpr!("axpy", E)(this.dim, cast(E)-1, src, this);
    }
    
    
    auto opBinary(string s : "*")(E src)
    {
        static if(argsType.length == 3)
            return vectorExpr!("gemv", E)(args[0] * src, args[1..3]);
        else
            return vectorExpr!("gemv", E)(args[0] * src, args[1..3], args[3] * src, args[4]);
    }
    
}