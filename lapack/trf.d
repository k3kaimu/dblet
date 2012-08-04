/** 行列をLU分解するためのモジュール
 * Authors: Kazuki Komatsu (k3kaimu)
 * License: NYSL
 */

module dblet.lapack.trf;

import std.functional;

static import dblas = scid.bindings.lapack.dlapack;
import dblet.matrix;
import dblet.traits;
import std.typecons;

version(unittest){
    import std.stdio;
    pragma(lib, "scid");
    pragma(lib, "blas");
    pragma(lib, "alice");
}

/**分解に失敗した場合に投げられる例外です
*/
class TrfException : Exception
{
    int info;
    this(int _info, string s, string fn = __FILE__, size_t ln = __LINE__){
        super(s, fn, ln);
        info = _info;
    }
}

struct LUPMatrix(T){
    Matrix!T lu;
    int[] p;
    
    this(Matrix!T a, int[] b){
        lu = a;
        p = b;
    }
}

auto lupFactorize(string how = "a == 0", M)(M a)
if(isMatrix!M)
{
    static if(isNormalMatrix!M)
        return getrf!(how, M.elementType)(a);
    else
        static assert(0);
}

/** m*nの一般マトリックスをA = PLUの形式に因子分解します。失敗した場合には例外を投げます$(BR)
例外の投げ方はテンプレートパラメータhowにより指定できます。$(BR)
1. "a == 0" :正常に終了(info == 0)しなければ例外を投げます$(BR)
2. "a >= 0" :正常に終了(info == 0)するか、因子分解は完了したがUは完全に特異である場合(info > 0)は例外は投げず、それ以外の場合(info < 0)は例外を投げます。$(BR)
3. "true" :たとえ失敗していたとしても常に結果を返すようにします。$(BR)
デフォルトでは1.の"info == 0"となっています。
*/
Tuple!(LUPMatrix!T, int) getrf(string how = "a == 0", T)(Matrix!T a)
if(isBlasType!T)
{
    auto _a = a.evalExpr;
    int[] Ipiv = new int[_a.rows];
    int info;
    dblas.getrf(_a.rows, _a.cols, _a.array.ptr, _a.rows, Ipiv.ptr, info);
    
    if(unaryFun!how(info))
        return tuple(LUPMatrix!T(_a, Ipiv), info);
        //return tuple(Ipiv, _a, info);
    else
        throw new TrfException(info, "Error : factorize to PLU.");
}
unittest{
    Matrix!double a = Matrix!double(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    auto plu = getrf(a);
    writeln(plu[0]);
    writeln(plu[1]);
}
