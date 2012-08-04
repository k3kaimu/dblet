/** Tupleを扱うためのライブラリ
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.tuple;

public import std.typecons;
static import ablas = dblet.blas.lv1;

auto tupleExpr(string Op, E, ArgT...)(ArgT src){
    return TupleExpr!(Op, E, ArgT)(src);
}

struct TupleExpr(string Op, E, ArgT...){
package:
    ArgT args;
    
public:
    alias Op operator;
    alias E elementType;
    alias ArgT argsType;
    alias typeof(this.evalExpr()) evalType;
    
    this(ArgT src){
        foreach(i, AType; ArgT)
            args[i] = src;
    }
    
    /*
    @property auto evalExpr(){
        mixin("return "~ Op ~ "_evalExpr();");
    }*/
    
    mixin TupleOpExpr!Op;
}

mixin template TupleOpExpr(string s : "rot"){
    
    auto opBinary(string s : "*", SType)(SType src)
    if(is(SType : E))
    {
        return tupleExpr!("rot", E)(this.args[0..2], this.args[3] * src, this.args[4] * src);
    }
    
    auto opOpAssign(string s : "*", SType)(SType src)
    if(is(SType : E))
    {
        this.args[3] *= src;
        this.args[4] *= src;
        return this;
    }
    
    @property auto evalExpr(){
        return ablas.rot(args);
    }
    
}

