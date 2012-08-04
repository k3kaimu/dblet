/** 逆行列を計算するためのモジュール
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.lapack.tri;

import std.typecons;
import std.typetuple;

static import dblas = scid.bindings.lapack.dlapack;
import dblet.matrix;
import dblet.traits;

import dblet.lapack.trf   : LUPMatrix;

private size_t _lwork;

static this(){
    _lwork = 64;
}

class TriException : Exception
{
    int info;
    this(int _info, string s, string fn = __FILE__, size_t ln = __LINE__){
        super(s, fn, ln);
        info = _info;
    }
}

Tuple!(Matrix!T, int) getri(string how = "info == 0", T)(LUPMatrix!T a)
if(isBlasType!T)
in{
    assert(a.lu.rows == a.lu.cols);
    assert(a.lu.rows != 0);
}
body{
    auto work = new T[a.lu.rows * _lwork];
    int info;
    auto _a = a.lu.dup;
    dblas.getri(_a.cols, _a.array.ptr, _a.rows, a.p.ptr, work.ptr, _lwork, info);
    _lwork = cast(uint)work[0];
    if(mixin(how))
        return tuple(_a, info);
    else
        throw new TriException(info, "Error : inverse matrix");
}
