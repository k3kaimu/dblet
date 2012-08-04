/** 連立一次方程式を解くモジュール
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.lapack.trs;

import std.functional   : unaryFun;
import std.typecons     : tuple, Tuple;

static import dblas = scid.bindings.lapack.dlapack;
import dblet.matrix;
import dblet.vector;
import dblet.traits;
import dblet.lapack.trf   : LUPMatrix;

class TrsException : Exception
{
    int info;
    this(int _info, string s, string fn = __FILE__, size_t ln = __LINE__){
        super(s, fn, ln);
        info = _info;
    }
}

auto 

Tuple!(Vector!T, int) getrs(string how = "a == 0", T)(LUPMatrix!T a, Vector!T b)
if(isBlasType!T)
{
    auto _b = b.dup;
    int info;
    dblas.getrs('N', _b.dim, 1, a.lu.array.ptr, a.lu.rows, a.p.ptr, _b.array.ptr, _b.dim, info);
    
    if(unaryFun!how(info))
        return tuple(_b, info);
    else
        throw new TrsException(info, "Error in Getrs");
}


